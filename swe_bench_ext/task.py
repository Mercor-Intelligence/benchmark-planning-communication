"""
SWE-Bench Extended Task Implementation.

This module provides the SweBenchExtTask class that extends BaseBenchmarkTask
from eval-framework to handle SWE-Bench-Ext specific task loading, prompts,
scripts, and result parsing.

Author: Mercor Intelligence
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from pydantic import Field

from lighthouse.core.benchmark_tasks.base_benchmark_task import (
    BaseBenchmarkTask,
    BaseTaskInstance,
)
from lighthouse.core.benchmark_tasks.models import (
    TestSummary,
    TestStatus,
    TaskStage,
    StageType,
    ArtifactSpec,
)
from lighthouse.core.benchmark_tasks.task_source import FolderTaskSource
from lighthouse.core.benchmark_tasks.benchmark_config import BenchmarkConfig
from lighthouse.core.benchmark_tasks.stage_postprocessing import StagePostProcessor
from lighthouse.core.post_eval.artifacts.postprocessor import ArtifactValidationPostProcessor
from lighthouse.common.parsing import (
    parse_test_output,
    normalize_test_id,
    get_framework_config,
    get_test_command_with_output,
)
from lighthouse.common.utils.cmd_generation import generate_git_init_script, generate_git_apply_script, generate_git_diff_script
from lighthouse.core.registry import benchmark_task
from lighthouse.core.grading.rubric.models import Rubric
from lighthouse.core.benchmark_tasks.task_source import TaskSource
from lighthouse.core.harness.continuation_policy import CompositeUserMessageHandler, ConstantFooter
from .config import SweBenchExtConfig


# =============================================================================
# Task Instance Model (extends BaseTaskInstance)
# =============================================================================

class SweBenchExtTaskInstance(BaseTaskInstance):
    """
    SWE-Bench-Ext specific task instance data.
    
    Extends the base task instance with SWE-Bench specific fields.
    """
    # Override image_uri to allow empty string initially (will be set by get_default_image_uri)
    image_uri: str = Field(default="", description="Docker image URI")
    
    # SWE-Bench specific fields
    language: str = Field(default="python", description="Programming language")
    test_framework: str = Field(default="pytest", description="Test framework")
    test_command: str = Field(default="", description="Command to run tests")
    test_files: List[str] = Field(default_factory=list, description="Test file paths")
    fail_to_pass: List[str] = Field(default_factory=list, description="Tests that should pass after fix")
    pass_to_pass: List[str] = Field(default_factory=list, description="Tests that should continue to pass")
    base_commit: str = Field(default="", description="Base commit hash for resetting test files")
    
    # Content fields
    problem_statement: str = Field(default="", description="Problem description")
    prompt_statement: str = Field(default="", description="Customized prompt (optional)")
    test_patch: str = Field(default="", description="Patch to apply for testing")
    golden_patch: str = Field(default="", description="Reference solution")
    requirements: List[str] = Field(default_factory=list, description="Requirements list")
    interface: str = Field(default="", description="Interface documentation")
    knowledge_base: str = Field(default="", description="Knowledge base documentation")


# =============================================================================
# Task Implementation
# =============================================================================

@benchmark_task("swe_bench_ext")
class SweBenchExtTask(BaseBenchmarkTask):
    """
    SWE-Bench Extended benchmark task.
    
    Extends BaseBenchmarkTask from eval-framework to handle:
    - Task loading from S3 or local filesystem
    - System/user prompt generation
    - Setup, grading, and test scripts
    - Test result parsing
    
    Usage:
        # From task source
        task = SweBenchExtTask.from_id("0xpolygon-bor-1710", task_source)
        
        # Get prompts
        system_prompt = task.get_system_prompt()
        user_prompt = task.get_initial_user_prompt()
        
        # Get scripts
        grading_scripts = task.generate_grading_setup_script()
        test_script = task.generate_test_run_script()
    """

    config: SweBenchExtConfig
    task_source: FolderTaskSource
    
    # === Class-level configuration ===
    config_class: ClassVar[Type[BenchmarkConfig]] = SweBenchExtConfig
    supported_task_sources: ClassVar[Tuple[Type, ...]] = (FolderTaskSource,)

    # === Instance attributes ===
    task_instance: SweBenchExtTaskInstance

    # =========================================================================
    # Multi-stage execution (config-driven)
    # =========================================================================

    def get_stages(self) -> List[TaskStage]:
        """
        Define stages for this task.

        Default: single solve stage.
        If config.enable_pr_artifacts is true: solve -> artifacts.
        """
        stage_tools = getattr(self.config, "stage_tools", {}) or {}
        solve_tools = stage_tools.get("solve", []) or []

        stages: List[TaskStage] = [
            TaskStage(
                id="solve",
                type=StageType.AGENT,
                name="Solve",
                tool_names=solve_tools,
                params={},
            )
        ]

        if getattr(self.config, "enable_pr_artifacts", False):
            out_dir = getattr(self.config, "artifacts_out_dir", "/workspace/repo/.agent_artifacts")
            artifacts_user_prompt = getattr(self.config, "artifacts_user_prompt", None)
            artifacts_tools = stage_tools.get("artifacts", []) or []
            specs = getattr(self.config, "artifacts_specs", None)
            # Defensive: accept dicts too (config YAML may come in as plain dicts)
            if specs and specs and not isinstance(specs[0], ArtifactSpec):
                try:
                    specs = [ArtifactSpec(**s) for s in specs]  # type: ignore[arg-type]
                except Exception:
                    specs = None
            stages.append(
                TaskStage(
                    id="artifacts",
                    type=StageType.AGENT,
                    name="Artifacts",
                    tool_names=artifacts_tools,
                    params={"out_dir": out_dir, "user_prompt": artifacts_user_prompt, "specs": specs},
                )
            )

        return stages

    def build_stage_system_prompt(self, stage: TaskStage, tool_prompts: Optional[List[str]] = None) -> str:
        """
        Optional per-stage system prompt override (via config.stage_system_prompts[stage.id]).
        Falls back to BaseBenchmarkTask behavior.
        """
        stage_system_prompts = getattr(self.config, "stage_system_prompts", {}) or {}
        override = (stage_system_prompts.get(stage.id) or "").strip()
        if override:
            prompt = override
            if tool_prompts:
                prompt += "\n\n" + "\n".join(tool_prompts)
            return prompt
        return super().build_stage_system_prompt(stage, tool_prompts=tool_prompts)

    def build_stage_user_prompt(self, stage: TaskStage, tool_prompts: Optional[List[str]] = None) -> str:
        """
        Optional per-stage user prompt override (via config.stage_user_prompts[stage.id]).
        For artifacts stage, uses hardcoded prompt (not from config).
        Falls back to BaseBenchmarkTask behavior.
        """
        stage_user_prompts = getattr(self.config, "stage_user_prompts", {}) or {}
        override = (stage_user_prompts.get(stage.id) or "").strip()
        
        # Hardcoded artifacts prompt (not from config)
        if not override and stage.id == "artifacts":
            override = """Now create documentation artifacts in {out_dir}:

1. Create PR_REPORT.md with:
   - ## Summary: Brief description of changes
   - ## Files Changed: List of modified files
   - ## Testing: How to test the changes
   - ## Notes: Any additional context

2. Create code_flow.puml with a PlantUML diagram showing the code flow of your implementation"""

        if override:
            # Best-effort templating for artifacts out_dir
            if stage.id == "artifacts":
                out_dir = (stage.params or {}).get("out_dir") or getattr(self.config, "artifacts_out_dir", "/workspace/repo/.agent_artifacts")
                try:
                    override = override.format(out_dir=out_dir)
                except Exception:
                    pass
            prompt = override
            if tool_prompts:
                prompt += "\n\n" + "\n".join(tool_prompts)
            return prompt

        return super().build_stage_user_prompt(stage, tool_prompts=tool_prompts)

    def get_user_message_handler(self, stage: TaskStage, options=None):
        """
        Build UserMessageHandler based on reminder policy configuration.
        
        Returns:
            CompositeUserMessageHandler with "once at start" reminder handler
            None if all reminders are off
        """
        from lighthouse.core.harness.continuation_policy import CompositeUserMessageHandler
        from swe_bench_ext.reminders import (
            README_REMINDER_LINE,
            ASK_QUESTION_REMINDER_LINE,
            ConstantReminderWithTiming,
        )
        
        # Get reminder policy from config
        policy = getattr(self.config, "reminder_policy", None)
        if not policy:
            return None
        
        # Normalize modes
        readme_mode = (policy.readme_mode or "off").strip().lower()
        ask_question_mode = (policy.ask_question_mode or "off").strip().lower()

        # Get tools enabled for this stage.
        # Fall back to options.tools (from CLI --tools/--add-tools) if stage doesn't specify tools.
        if hasattr(stage, "tool_names") and stage.tool_names:
            stage_tool_names = stage.tool_names
        elif options is not None and hasattr(options, "tools") and options.tools is not None:
            stage_tool_names = options.tools
        else:
            stage_tool_names = []
        
        # =====================================================================
        # PHASE 1: Build constant reminder items (constant mode implementation)
        # =====================================================================
        constant_items = []
        
        # README reminder: always shown if mode is "constant"
        if readme_mode == "constant":
            constant_items.append(README_REMINDER_LINE)
        
        # ask_question reminder: shown when mode is "constant" AND the tool is available.
        if ask_question_mode == "constant":
            if "ask_question" in stage_tool_names:
                constant_items.append(ASK_QUESTION_REMINDER_LINE)
        
        # If no reminders enabled, return None 
        if not constant_items:
            return None
        
        # =====================================================================
        # Create timing-aware handler 
        # =====================================================================
        # This handler checks for assistant_messages before showing reminders
        # Default to "every turn" reminders.
        timing_handler = ConstantReminderWithTiming(reminder_items=constant_items, frequency=1)
        
        # =====================================================================
        # Return CompositeUserMessageHandler
        # =====================================================================
        # For Phase 1 (constant mode):
        # - handlers=[timing_handler] contains our timing-aware constant reminder
        #
        # For Phase 2 (staged mode), we would:
        # - Add staged handler instances to the handlers=[] list
        # - All handlers check conversation state before returning messages
        return CompositeUserMessageHandler(
            handlers=[timing_handler],  # Timing-aware constant reminder handler
            constant_footer=None,  # Not needed - handler controls display
            emit_footer_when_no_message=False,  # Handler controls timing
        )

    def get_stage_postprocessors(self, stage: TaskStage) -> List[StagePostProcessor]:
        """
        Return postprocessors for the given stage.
        
        For the artifacts stage, returns ArtifactValidationPostProcessor if enabled.
        """
        if stage.id == "artifacts" and getattr(self.config, "enable_pr_artifacts", False):
            return [ArtifactValidationPostProcessor.from_stage(stage)]
        return []
    
    # =========================================================================
    # Task Loading (implements abstract method)
    # =========================================================================
    
    @classmethod
    def _load_task(cls, task_id: str, task_source: FolderTaskSource) -> SweBenchExtTaskInstance:
        """
        Load task data from the task source.
        
        Implements BaseBenchmarkTask._load_task()
        
        Args:
            task_id: Task identifier
            task_source: Source to load from (FolderTaskSource, S3, etc.)
            
        Returns:
            SweBenchExtTaskInstance with loaded data
        """
        # Load test_metadata.json
        
        try:
            metadata_content = task_source.get_task_file_contents(task_id, "test_metadata.json")
            metadata = json.loads(metadata_content)
        except Exception:
            metadata = {}
        
        # Helper to load file with fallback
        def load_file(filename: str, default: str = "") -> str:
            try:
                return task_source.get_task_file_contents(task_id, filename)
            except Exception:
                return default
        
        # Load requirements.json
        try:
            req_content = task_source.get_task_file_contents(task_id, "requirements.json")
            requirements = json.loads(req_content)
        except Exception:
            requirements = []
        
        # Load problem statement
        problem_statement = load_file("problem_statement.md")
        
        return SweBenchExtTaskInstance(
            id=task_id,
            image_uri="",  # Will be set by get_default_image_uri
            language=metadata.get("language", "python"),
            test_framework=metadata.get("test_framework", metadata.get("language", "pytest")),
            test_command=metadata.get("test_command", ""),
            test_files=metadata.get("test_files", []),
            fail_to_pass=metadata.get("FAIL_TO_PASS", metadata.get("fail_to_pass", [])),
            pass_to_pass=metadata.get("PASS_TO_PASS", metadata.get("pass_to_pass", [])),
            base_commit=metadata.get("base_commit", ""),
            problem_statement=problem_statement,
            prompt_statement=load_file("prompt_statement.md", problem_statement),
            test_patch=load_file("test.patch"),
            golden_patch=load_file("golden.patch"),
            requirements=requirements,
            interface=load_file("interface.md"),
            knowledge_base=load_file("knowledge_base.md"),
        )
    
    @classmethod
    def from_id(
        cls,
        task_id: str,
        task_source: FolderTaskSource,
        image_uri_override: Optional[str] = None,
        **config_overrides,
    ) -> "SweBenchExtTask":
        """
        Get the task for a given task ID with rubric grader attached.
        
        Overrides base class to attach SweBenchExtRubricGrader.
        
        Args:
            task_id: The unique identifier for the task
            task_source: The task source to load from
            image_uri_override: Optional image URI override
            **config_overrides: Optional config field overrides
            
        Returns:
            Initialized task instance with rubric grader attached
        """
        # Call parent implementation
        task_instance = cls._load_task(task_id, task_source)
        task = cls(task_instance=task_instance, task_source=task_source, **config_overrides)

        if not task_instance.image_uri:
            task_instance.image_uri = image_uri_override if image_uri_override else task.get_default_image_uri()

        task_source.build_docker_image_if_not_exists(task_id, task_instance.image_uri)
        
        # Attach rubric grader if rubric exists
        # Note: We only create the grader instance here. The actual rubric loading
        # happens lazily when init_rubric_grader() is called from the harness.
        try:
            from .rubric_grader import SweBenchExtRubricGrader
            
            # Use configurable rubric file path
            rubric_file = task.config.rubric_file if task.config else "rubric/rubric.json"
            task_source.get_task_file_contents(task_id, rubric_file)
            
            # Create grader instance without loading rubric yet
            # The rubric will be loaded by init_rubric_grader() when needed
            task.rubric_grader = SweBenchExtRubricGrader()
            
        except Exception:
            # No rubric or failed to load - that's OK, rubric grading is optional
            pass

        return task
    
    def _get_rubric_dict(self) -> Dict[str, Any]:
        """
        Get the rubric dictionary for this task.
        
        Returns:
            Rubric dictionary loaded from task source
            
        Raises:
            ValueError: If rubric cannot be loaded
        """
        try:
            rubric_file = self.config.rubric_file if self.config else "rubric/rubric.json"
            rubric_content = self.task_source.get_task_file_contents(
                self.task_instance.id,
                rubric_file
            )
            return json.loads(rubric_content)
        except Exception as e:
            raise ValueError(f"Failed to load rubric for task {self.task_instance.id}: {e}")
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def get_default_image_uri(self) -> str:
        """Get Docker image URI for this task.
        
        Priority:
        1. If task_instance.image_uri is already set (by from_id with image_uri_override), use it
        2. If config has image_uri_template, use config.get_image_uri()
        3. Otherwise, use default local image name
        """
        # Check if image_uri was already set (e.g., via image_uri_override in from_id)
        if self.task_instance.image_uri:
            return self.task_instance.image_uri
        
        task_id = self.task_instance.id
        if self.config:
            return self.config.get_image_uri(task_id)
        return f"swe-bench-ext-{task_id}:latest"
    
    def get_golden_solution(self) -> str:
        """Get the golden patch for this task."""
        return self.task_instance.golden_patch
    
    def get_system_prompt(self, tool_prompts: Optional[List[str]] = None) -> str:
        """Generate system prompt for the agent."""
        inst = self.task_instance
        
        prompt = f"""You are an expert software engineer tasked with fixing a bug in a codebase.

LANGUAGE: {inst.language}
TEST FRAMEWORK: {inst.test_framework}
WORKING DIRECTORY: {self.workdir}

INSTRUCTIONS:
1. Analyze the problem statement carefully
2. Explore the codebase to understand the issue
3. Make minimal, focused changes to fix the bug
4. Do NOT modify test files unless explicitly required
5. Ensure your changes don't break existing functionality

When you're done, your solution will be tested automatically against the test suite.
Make sure your changes are complete and the code compiles/runs correctly."""
                
        if tool_prompts:
            prompt += "\n\n" + "\n".join(tool_prompts)
        
        return prompt
    
    def get_initial_user_prompt(
        self,
        tool_prompts: Optional[List[str]] = None,
    ) -> str:
        """Generate initial user message with the problem.
        
        Args:
            tool_prompts: Optional list of tool-specific prompts to append.
            excluded_context: List of context sections to exclude.
                Valid values: ["problem_statement", "requirements", "interface"].
                
        Returns:
            Formatted prompt string with only the provided sections.
        """
        inst = self.task_instance
        excluded_context = self.config.excluded_context or []
        
        parts = ["Please solve the following coding issue:\n"]
        
        # Problem statement (using prompt_statement which may be customized)
        if inst.problem_statement and "problem_statement" not in excluded_context:
            parts.append(f"## Problem Description\n{inst.problem_statement}\n")
        
        # User request (always included)
        parts.append(f"## User Request\n{inst.prompt_statement}\n")
        
        # Requirements (formatted as numbered list)
        if inst.requirements and "requirements" not in excluded_context:
            requirements_text = "\n".join(
                [f"{i + 1}. {req}" for i, req in enumerate(inst.requirements)]
            )
            parts.append(f"## Requirements\n{requirements_text}\n")
        
        # Interface specifications
        if inst.interface and "interface" not in excluded_context:
            parts.append(f"## Interface Specifications\n{inst.interface}\n")
        
        parts.append(
            "Please implement a solution that satisfies all the requirements above."
        )
        
        prompt = "\n".join(parts)
        
        return prompt
    
    def generate_setup_script(self) -> Union[str, List[str]]:
        """Generate setup scripts (run before agent starts).
        
        This script:
        1. Marks workspace as safe directory for git
        2. Initializes a git repo if it doesn't exist
        3. Configures git user if not set (required for commits)
        4. Creates an initial commit if no commits exist
        5. Removes excluded context files from /workspace/ if configured
        """
        workdir = self.workdir
        
        # Get git init script and extract the body (skip shebang and set -e lines)
        git_init_script = generate_git_init_script(workdir)
        git_init_body = '\n'.join(git_init_script.split('\n')[2:])
        
        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Mark workspace as safe directory",
            f"git config --global --add safe.directory {workdir}",
            git_init_body,
        ]
        
        # Remove excluded context files from /workspace/
        # This prevents the agent from accessing these files directly
        excluded_context = self.config.excluded_context if self.config else []
        if excluded_context:
            # Mapping from context names to file names
            context_to_file = {
                "problem_statement": "problem_statement.md",
                "requirements": "requirements.json",
                "interface": "interface.md",
                "knowledge_base": "knowledge_base.md",
            }
            
            script_parts.append("")
            script_parts.append("# Remove excluded context files from /workspace/")
            
            for context_name in excluded_context:
                if context_name in context_to_file:
                    filename = context_to_file[context_name]
                    script_parts.append(f'rm -f "/workspace/{filename}"')
        
        return "\n".join(script_parts)
    
    def generate_grading_setup_script(
        self,
        apply_golden_patch: bool = False,
    ) -> Union[str, List[str]]:
        """Generate grading setup scripts.
        
        This script:
        1. Configures git and marks the directory as safe
        2. Resets test files to the base commit state (or initial commit if base_commit unavailable)
        3. Creates the test patch file from embedded base64 content
        4. Optionally applies the golden patch (for golden patch testing mode)
        
        Args:
            apply_golden_patch: If True, also apply the golden patch before tests.
        """
        inst = self.task_instance
        workdir = self.workdir
        
        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Configure git",
            'git config --global user.email "grader@swebench.ext"',
            'git config --global user.name "Grader"',
            f"git config --global --add safe.directory {workdir}",
            "",
            f"cd {workdir}",
            "",
            "# Determine which commit to use for resetting test files",
            "# Prefer base_commit if provided and valid, otherwise use initial commit",
        ]
        
        # Add base commit determination logic
        if inst.base_commit:
            script_parts.append(f'''
BASE_COMMIT="{inst.base_commit}"
if git cat-file -e "$BASE_COMMIT^{{commit}}" 2>/dev/null; then
    reset_commit="$BASE_COMMIT"
    echo "Using base commit: $reset_commit"
else
    reset_commit=$(git rev-list --max-parents=0 HEAD)
    echo "Base commit not in history, using initial commit: $reset_commit"
fi
''')
        else:
            script_parts.append('''
reset_commit=$(git rev-list --max-parents=0 HEAD)
echo "No base commit specified, using initial commit: $reset_commit"
''')
        
        # Reset test files to the determined commit state (undo any agent modifications to test files)
        if inst.test_files:
            script_parts.append("# Reset test files to the determined commit state")
            script_parts.append("# This ensures we evaluate the agent's code changes, not test modifications")
            
            for test_file in inst.test_files:
                # Check if file existed in reset commit; if so restore it, else delete it
                script_parts.append(f'''
if git cat-file -e "$reset_commit:{test_file}" 2>/dev/null; then
    git checkout "$reset_commit" -- "{test_file}" || true
else
    # File didn't exist in reset commit, delete it if it exists now
    if [ -f "{test_file}" ]; then
        rm "{test_file}"
    fi
fi''')
            
            script_parts.append("")
        
        # Apply golden patch if requested (for golden patch testing mode)
        if apply_golden_patch and inst.golden_patch:
            script_parts.append("# Apply golden patch")
            script_parts.append('echo "=== Applying golden patch ==="')
            # Use generate_git_apply_script and extract body (skip shebang and set -e)
            golden_apply_script = generate_git_apply_script(inst.golden_patch, workdir)
            golden_apply_body = '\n'.join(golden_apply_script.split('\n')[2:])
            script_parts.append(golden_apply_body)
            script_parts.append('echo "Golden patch applied successfully"')
        
        # Create and apply test patch
        if inst.test_patch:
            script_parts.append("# Apply test patch")
            script_parts.append('echo "=== Applying test patch ==="')
            # Use generate_git_apply_script with allow_incomplete=True to use --reject mode
            # This allows partial application and creates .rej files for failed hunks
            test_apply_script = generate_git_apply_script(inst.test_patch, workdir, allow_incomplete=True)
            test_apply_lines = test_apply_script.split('\n')[2:]  # Skip shebang and set -e
            # Add || true to git apply lines to prevent set -e from exiting the script
            # when patches fail to apply (git apply --reject still returns non-zero on failures)
            test_apply_lines = [
                line + ' || true' if 'git apply' in line else line
                for line in test_apply_lines
            ]
            script_parts.append('\n'.join(test_apply_lines))
            
            # Report any failed patches to stderr
            script_parts.append('''
# Check for .rej files (hunks that failed to apply) and report to stderr
rej_files=$(find . -name "*.rej" -type f 2>/dev/null)
if [ -n "$rej_files" ]; then
    echo "WARNING: Some patch hunks failed to apply:" >&2
    for rej_file in $rej_files; do
        # Get the original file name by removing .rej suffix
        original_file="${rej_file%.rej}"
        echo "  - Failed to patch: $original_file (see $rej_file for rejected hunks)" >&2
    done
fi''')
            script_parts.append('echo "Test patch applied"')
        else:
            script_parts.append("# No test patch to apply")
            script_parts.append('echo "No test patch to apply"')
        
        return "\n".join(script_parts)
    
    
    def generate_test_run_script(self) -> Union[str, List[str]]:
        """Generate test execution script.
        
        This script:
        1. Changes to the working directory
        2. Creates necessary output directories
        3. Runs the test command with appropriate output flags
        4. If a result file is generated, outputs it to stdout with markers
        
        The script ensures that structured test output (JSON, XML, etc.) is available
        in stdout/stderr for parsing, even if the framework writes to a file.
        """
        inst = self.task_instance
        workdir = self.workdir
        
        # Get the test command with output flags
        base_command = inst.test_command
        if not base_command:
            # Default commands per test framework
            default_commands = {
                "pytest": "pytest -xvs",
                "go": "go test -v ./...",
                "jest": "npm test",
                "vitest": "npm test",
                "cargo": "cargo test",
                "cargo-nextest": "cargo nextest run",
                "maven": "mvn test",
                "gradle": "gradle test",
            }
            base_command = default_commands.get(inst.test_framework, "echo 'No test command specified'")
        
        test_cmd = get_test_command_with_output(base_command, inst.test_framework)
        
        # Get framework config for result file location
        config = get_framework_config(inst.test_framework, base_command)
        result_file = config.get("result_file")
        
        script_parts = [
            "#!/bin/bash",
            "# Don't use set -e - we want to capture test failures, not abort on them",
            "set -o pipefail",
            "",
            f"cd {workdir}",
            "",
            "# Create test results directory",
            "mkdir -p /workspace/test-results",
            "",
            "# Markers for parsing (helps extract test output from other stdout content)",
            'echo "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>"',
            "",
            "# Run tests",
            test_cmd,
            "test_exit_code=$?",
            "",
        ]
        
        # If there's a result file, output its contents for parsing
        if result_file:
            if "*" in result_file:
                # Handle glob patterns (e.g., Maven/JUnit XMLs)
                script_parts.append(f'''
# Output result files (glob pattern: {result_file})
echo "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
for f in {result_file}; do
    if [ -f "$f" ]; then
        echo "=== FILE: $f ==="
        cat "$f"
        echo ""
    fi
done 2>/dev/null || true
echo "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"
''')
            else:
                # Single result file
                script_parts.append(f'''
# Output result file: {result_file}
echo "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
if [ -f "{result_file}" ]; then
    cat "{result_file}"
fi
echo "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"
''')
        
        script_parts.append('''
echo "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>"

# Exit with test exit code
exit $test_exit_code
''')
        
        return "\n".join(script_parts)
    
    def parse_test_results(self, test_output: str) -> TestSummary:
        """Parse test output into structured results.
        
        This method:
        1. Extracts structured test output from markers if present
        2. Uses the appropriate parser for the test framework
        3. Normalizes test IDs for stable matching
        4. Matches against expected FAIL_TO_PASS and PASS_TO_PASS lists
        5. Computes overall pass/fail and score
        
        Args:
            test_output: Raw test output from running the test script.
            
        Returns:
            TestSummary with parsed results, matching, and computed score.
        """
        inst = self.task_instance
        test_framework = inst.test_framework
        
        # Normalize expected test IDs
        fail_to_pass = [normalize_test_id(tid, test_framework) for tid in inst.fail_to_pass]
        pass_to_pass = [normalize_test_id(tid, test_framework) for tid in inst.pass_to_pass]
        
        # Parse test output using the common parsing function
        parsed_results = self._parse_raw_test_output(test_output, test_framework)
        
        # Handle case where parsing failed completely
        if parsed_results is None:
            parsed_results = {}
        
        # Handle synthetic build/compile tests
        # If a test ID ends with ::build or ::compile and is NOT in parsed_results,
        # it means the build/compilation SUCCEEDED (failures would be reported)
        for tid in fail_to_pass + pass_to_pass:
            if (tid.endswith("::build") or tid.endswith("::compile")) and tid not in parsed_results:
                parsed_results[tid] = "PASSED"
        
        # Identify packages that failed to build (package-level failures without ::)
        build_failed_packages = {
            pkg
            for pkg, status in parsed_results.items()
            if status == "FAILED" and "::" not in pkg
        }
        
        # Match FAIL_TO_PASS tests
        fail_to_pass_results = {}
        for test_id in fail_to_pass:
            fail_to_pass_results[test_id] = self._match_test_with_fuzzy(
                test_id, parsed_results, build_failed_packages
            )
        
        # Match PASS_TO_PASS tests
        pass_to_pass_results = {}
        for test_id in pass_to_pass:
            pass_to_pass_results[test_id] = self._match_test_with_fuzzy(
                test_id, parsed_results, build_failed_packages
            )
        
        # Compute pass/fail
        all_f2p_passed = all(v == "PASSED" for v in fail_to_pass_results.values())
        all_p2p_passed = all(v == "PASSED" for v in pass_to_pass_results.values())
        passed = all_f2p_passed and all_p2p_passed
        
        # Convert parsed results to TestStatus enum
        test_statuses = {}
        for test_id, status in parsed_results.items():
            if status == "PASSED":
                test_statuses[test_id] = TestStatus.PASS
            elif status == "FAILED":
                test_statuses[test_id] = TestStatus.FAIL
            elif status == "SKIPPED":
                test_statuses[test_id] = TestStatus.SKIP
            else:
                test_statuses[test_id] = TestStatus.UNKNOWN
        
        return TestSummary(
            score=1.0 if passed else 0.0,
            test_statuses=test_statuses,
            metadata={
                "fail_to_pass_results": fail_to_pass_results,
                "pass_to_pass_results": pass_to_pass_results,
                "f2p_passed": sum(1 for v in fail_to_pass_results.values() if v == "PASSED"),
                "f2p_total": len(fail_to_pass_results),
                "p2p_passed": sum(1 for v in pass_to_pass_results.values() if v == "PASSED"),
                "p2p_total": len(pass_to_pass_results),
            },
        )
    
    def _extract_result_file_content(self, test_output: str) -> Optional[str]:
        """Extract result file content from test output if markers are present.
        
        Args:
            test_output: Full test output including potential result file content.
            
        Returns:
            Extracted result file content, or None if not found.
        """
        start_marker = "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
        end_marker = "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"
        
        if start_marker in test_output and end_marker in test_output:
            start_idx = test_output.find(start_marker) + len(start_marker)
            end_idx = test_output.find(end_marker)
            if start_idx < end_idx:
                return test_output[start_idx:end_idx].strip()
        
        return None
    
    def _parse_raw_test_output(self, test_output: str, test_framework: str) -> Optional[Dict[str, str]]:
        """Parse raw test output and return normalized test results.
        
        This function:
        1. Extracts structured test output from markers if present
        2. Uses the appropriate parser for the test framework
        3. Falls back to parsing full output if result file parsing fails
        4. Normalizes test IDs to remove unstable runtime prefixes
        
        Args:
            test_output: Raw test output string.
            test_framework: Test framework name (e.g., 'pytest', 'go', 'jest').
            
        Returns:
            Dict of normalized test_id -> status, or None if parsing failed completely.
        """
        # Try to extract result file content from markers
        result_file_content = self._extract_result_file_content(test_output)
        
        # Parse test output using appropriate parser
        if result_file_content:
            parsed_results = parse_test_output(result_file_content, test_framework)
            # Fallback to full output if result file parsing failed
            if not parsed_results:
                parsed_results = parse_test_output(test_output, test_framework)
        else:
            parsed_results = parse_test_output(test_output, test_framework)
        
        # Normalize test IDs to remove unstable runtime prefixes
        if parsed_results:
            parsed_results = {
                normalize_test_id(tid, test_framework): status
                for tid, status in parsed_results.items()
            }
        
        return parsed_results
    
    def _match_test_with_fuzzy(
        self,
        test_id: str,
        parsed_results: Dict[str, str],
        build_failed_packages: set,
    ) -> str:
        """Try to match a test ID against parsed results, using fuzzy matching if needed.
        
        Args:
            test_id: The expected test ID to find.
            parsed_results: Dict of parsed test IDs to statuses.
            build_failed_packages: Set of packages that failed to build.
            
        Returns:
            Test status: "PASSED", "FAILED", "SKIPPED", or "NOT_FOUND".
        """
        # Direct match
        if test_id in parsed_results:
            return parsed_results[test_id]
        
        # Try normalized matching (handles different delimiters, file extensions, etc.)
        normalized_test_id = normalize_test_id(test_id)
        for parsed_id, status in parsed_results.items():
            if normalize_test_id(parsed_id) == normalized_test_id:
                return status
        
        # Try fuzzy matching for tests with path::name format
        if "::" in test_id:
            file_path, test_name = test_id.rsplit("::", 1)
            
            # For parameterized tests, extract base name before brackets
            base_test_name = test_name.split("[")[0] if "[" in test_name else test_name
            
            # Normalize file_path for comparison
            normalized_file_path = normalize_test_id(file_path)
            
            # Look for tests in same file with similar names
            for parsed_id in parsed_results.keys():
                if "::" in parsed_id:
                    parsed_file_path, parsed_test_name = parsed_id.rsplit("::", 1)
                    
                    # Check if file paths match (with normalization)
                    if normalize_test_id(parsed_file_path) == normalized_file_path:
                        parsed_base_name = (
                            parsed_test_name.split("[")[0]
                            if "[" in parsed_test_name
                            else parsed_test_name
                        )
                        
                        # Exact match on base name (handles parameterized tests)
                        if base_test_name == parsed_base_name:
                            return parsed_results[parsed_id]
                        
                        # Fuzzy match on words for renamed tests
                        test_words = set(test_name.lower().split())
                        parsed_words = set(parsed_test_name.lower().split())
                        if len(test_words & parsed_words) >= len(test_words) * 0.7:
                            return parsed_results[parsed_id]
        
        # Check if the test's package failed to build
        test_package = test_id.split("::")[0] if "::" in test_id else test_id
        if test_package in build_failed_packages:
            return "FAILED"
        
        return "NOT_FOUND"
    
    def generate_grading_explanation(
        self,
        test_summary: TestSummary,
        test_output: str = "",
        max_output_length: int = 100000,
    ) -> str:
        """Generate a human-readable explanation of test results.
        
        Args:
            test_summary: TestSummary from parse_test_results.
            test_output: Optional raw test output to include.
            max_output_length: Maximum length of test output to include.
            
        Returns:
            Formatted explanation string.
        """
        metadata = test_summary.metadata
        f2p_passed = metadata.get("f2p_passed", 0)
        f2p_total = metadata.get("f2p_total", 0)
        p2p_passed = metadata.get("p2p_passed", 0)
        p2p_total = metadata.get("p2p_total", 0)
        fail_to_pass_results = metadata.get("fail_to_pass_results", {})
        pass_to_pass_results = metadata.get("pass_to_pass_results", {})
        
        parts = [
            "Test Results:",
            f"  FAIL_TO_PASS: {f2p_passed}/{f2p_total} passed",
            f"  PASS_TO_PASS: {p2p_passed}/{p2p_total} passed",
            f"  Total parsed tests: {len(test_summary.test_statuses)}",
        ]
        
        # Get failed tests
        failed_f2p = [t for t, v in fail_to_pass_results.items() if v != "PASSED"]
        failed_p2p = [t for t, v in pass_to_pass_results.items() if v != "PASSED"]
        
        # Show parsed tests when there are failures (for debugging)
        if (failed_f2p or failed_p2p) and len(test_summary.test_statuses) <= 100:
            parts.append("\nParsed test IDs:")
            for test_id in sorted(test_summary.test_statuses.keys()):
                parts.append(f"  {test_id}: {test_summary.test_statuses[test_id].value}")
        
        if failed_f2p:
            parts.append(f"\nFailed FAIL_TO_PASS tests ({len(failed_f2p)}):")
            for test in failed_f2p[:10]:
                parts.append(f"  - {test}: {fail_to_pass_results[test]}")
            if len(failed_f2p) > 10:
                parts.append(f"  ... and {len(failed_f2p) - 10} more")
        
        if failed_p2p:
            parts.append(f"\nFailed PASS_TO_PASS tests ({len(failed_p2p)}):")
            for test in failed_p2p[:10]:
                parts.append(f"  - {test}: {pass_to_pass_results[test]}")
            if len(failed_p2p) > 10:
                parts.append(f"  ... and {len(failed_p2p) - 10} more")
        
        if test_output:
            truncated = test_output[:max_output_length]
            if len(test_output) > max_output_length:
                truncated += "\n... (output truncated)"
            parts.append(f"\nTest output:\n{truncated}")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Metadata
    # =========================================================================
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get task metadata for logging/storage."""
        inst = self.task_instance
        return {
            "task_id": inst.id,
            "language": inst.language,
            "test_framework": inst.test_framework,
            "test_command": inst.test_command,
            "num_fail_to_pass": len(inst.fail_to_pass),
            "num_pass_to_pass": len(inst.pass_to_pass),
            "num_test_files": len(inst.test_files),
            "test_files": inst.test_files,
            "has_golden_patch": bool(inst.golden_patch),
            "has_interface": bool(inst.interface),
            "has_requirements": bool(inst.requirements),
            "has_knowledge_base": bool(inst.knowledge_base),
            "base_commit": inst.base_commit,
            "FAIL_TO_PASS": inst.fail_to_pass,
            "PASS_TO_PASS": inst.pass_to_pass,
        }
        
    def get_grading_guidelines(self) -> str:
        """
        Get grading guidelines for SWE-Bench-Ext tasks.
        
        Returns embedded guidelines (can be overridden to load from file).
        """
        return """
# SWE-Bench Grading Guidelines

- Focus on code changes (git diff) as primary evidence
- Passing tests strongly indicate correctness
- Give credit for reasonable attempts even with minor issues
- Be strict about missing functionality
- Style issues should not heavily penalize functional solutions
        """.strip()
    
    def __repr__(self) -> str:
        inst = self.task_instance
        return (
            f"SweBenchExtTask("
            f"id='{inst.id}', "
            f"language='{inst.language}', "
            f"test_framework='{inst.test_framework}'"
            f")"
        )

    def generate_solution_fetch_script(self) -> Union[str, List[str]]:
        """
        Generate script to fetch the solution from the sandbox.
        
        Returns:
            Bash script that outputs the solution to stdout
        """
        return generate_git_diff_script(target_dir=self.workdir, commit=self.task_instance.base_commit or "$(git rev-list --max-parents=0 HEAD)")
