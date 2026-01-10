"""
SWE-Bench Extended Task Implementation.

This module provides the SweBenchExtTask class that extends BaseBenchmarkTask
from eval-framework to handle SWE-Bench-Ext specific task loading, prompts,
scripts, and result parsing.

Author: Mercor Intelligence
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING

# Add eval-framework to path if running as submodule
_eval_framework_path = Path(__file__).parent.parent / "eval-framework"
if _eval_framework_path.exists() and str(_eval_framework_path) not in sys.path:
    sys.path.insert(0, str(_eval_framework_path))

# Import from eval-framework (graceful fallback for standalone testing)
try:
    from core.benchmark_tasks.base_benchmark_task import BaseBenchmarkTask
    from core.benchmark_tasks.models import TestSummary, TestStatus
    from core.benchmark_tasks.task_source import TaskSource, FolderTaskSource, S3ZipTaskSource
    from core.benchmark_tasks.benchmark_config import BenchmarkConfig, BenchmarkType
    from core.parsing import create_test_summary_from_output
    from core.registry import benchmark_task
    
    EVAL_FRAMEWORK_AVAILABLE = True
except ImportError:
    EVAL_FRAMEWORK_AVAILABLE = False
    BaseBenchmarkTask = object
    benchmark_task = lambda name: lambda cls: cls

from .config import SweBenchExtConfig


# Register with eval-framework
@benchmark_task("swe_bench_ext")
class SweBenchExtTask(BaseBenchmarkTask):
    """
    SWE-Bench Extended benchmark task.
    
    Extends BaseBenchmarkTask from eval-framework to handle:
    - Task loading from S3 or local filesystem
    - System/user prompt generation
    - Setup, grading, and test scripts
    - Test result parsing
    
    Task data structure (from S3 zip):
        {task_id}/
        ├── test_metadata.json      # Test framework, command, fail_to_pass, etc.
        ├── problem_statement.md    # Problem description
        ├── prompt_statement.md     # (optional) Customized prompt
        ├── test.patch              # Patch to apply for testing
        ├── golden.patch            # (optional) Reference solution
        ├── requirements.json       # (optional) List of requirements
        ├── interface.md            # (optional) Interface documentation
        └── Dockerfile              # Docker image definition
    """
    
    # === Class-level configuration ===
    config_class: ClassVar[Type[BenchmarkConfig]] = SweBenchExtConfig
    supported_task_sources: ClassVar[Tuple[Type, ...]] = (FolderTaskSource, S3ZipTaskSource) if EVAL_FRAMEWORK_AVAILABLE else ()
    
    # === Instance attributes (loaded from task source) ===
    language: str = "python"
    test_framework: str = "pytest"
    test_command: str = ""
    test_files: List[str] = None
    fail_to_pass: List[str] = None
    pass_to_pass: List[str] = None
    problem_statement: str = ""
    prompt_statement: str = ""
    test_patch: str = ""
    golden_patch: str = ""
    requirements: List[str] = None
    interface: str = ""
    
    def __init__(
        self,
        task_id: str,
        task_source: "TaskSource" = None,
        **config_overrides,
    ):
        """
        Initialize a SWE-Bench-Ext task.
        
        Args:
            task_id: Task identifier
            task_source: TaskSource to load from
            **config_overrides: Config field overrides
        """
        # Initialize lists
        self.test_files = []
        self.fail_to_pass = []
        self.pass_to_pass = []
        self.requirements = []
        
        # Call parent init if available
        if EVAL_FRAMEWORK_AVAILABLE and task_source is not None:
            super().__init__(task_id=task_id, task_source=task_source, **config_overrides)
        else:
            self.task_id = task_id
            self.config = SweBenchExtConfig(**config_overrides) if config_overrides else SweBenchExtConfig()
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_dict(cls, task_id: str, data: Dict[str, Any], **config_overrides) -> "SweBenchExtTask":
        """
        Create a task from a dictionary.
        
        Args:
            task_id: Task identifier
            data: Dictionary with task data
            **config_overrides: Config overrides
            
        Returns:
            SweBenchExtTask instance
        """
        task = cls(task_id=task_id, **config_overrides)
        
        # Load from data
        task.language = data.get("language", "python")
        task.test_framework = data.get("test_framework", "pytest")
        task.test_command = data.get("test_command", "")
        task.test_files = data.get("test_files", [])
        task.fail_to_pass = data.get("FAIL_TO_PASS", data.get("fail_to_pass", []))
        task.pass_to_pass = data.get("PASS_TO_PASS", data.get("pass_to_pass", []))
        task.problem_statement = data.get("problem_statement", "")
        task.prompt_statement = data.get("prompt_statement", "")
        task.test_patch = data.get("test_patch", "")
        task.golden_patch = data.get("golden_patch", "")
        task.requirements = data.get("requirements", [])
        task.interface = data.get("interface", "")
        task.image_uri = task.get_default_image_uri()
        
        return task
    
    # =========================================================================
    # Task Loading (implements abstract method)
    # =========================================================================
    
    def _load_task(self, task_source: "TaskSource") -> None:
        """
        Load task data from the task source.
        
        Implements BaseBenchmarkTask._load_task()
        """
        # Load test_metadata.json
        try:
            metadata_content = task_source.get_task_file_contents(self.task_id, "test_metadata.json")
            metadata = json.loads(metadata_content)
        except Exception:
            metadata = {}
        
        # Load from metadata
        self.language = metadata.get("language", "python")
        self.test_framework = metadata.get("test_framework", "pytest")
        self.test_command = metadata.get("test_command", "")
        self.test_files = metadata.get("test_files", [])
        self.fail_to_pass = metadata.get("FAIL_TO_PASS", metadata.get("fail_to_pass", []))
        self.pass_to_pass = metadata.get("PASS_TO_PASS", metadata.get("pass_to_pass", []))
        
        # Load text files
        self.problem_statement = self._load_file(task_source, "problem_statement.md", "")
        self.prompt_statement = self._load_file(task_source, "prompt_statement.md", self.problem_statement)
        self.test_patch = self._load_file(task_source, "test.patch", "")
        self.golden_patch = self._load_file(task_source, "golden.patch", "")
        self.interface = self._load_file(task_source, "interface.md", "")
        
        # Load requirements
        try:
            req_content = task_source.get_task_file_contents(self.task_id, "requirements.json")
            self.requirements = json.loads(req_content)
        except Exception:
            self.requirements = []
        
        # Set image URI
        self.image_uri = self.get_default_image_uri()
    
    def _load_file(self, task_source: "TaskSource", filename: str, default: str = "") -> str:
        """Load a file from task source with fallback."""
        try:
            return task_source.get_task_file_contents(self.task_id, filename)
        except Exception:
            return default
    
    @classmethod
    def from_local_path(cls, task_dir: Path, **config_overrides) -> "SweBenchExtTask":
        """
        Load a task from a local directory.
        
        Convenience method for local development/testing.
        
        Args:
            task_dir: Path to task directory
            **config_overrides: Config overrides
            
        Returns:
            SweBenchExtTask instance
        """
        task_id = task_dir.name
        task = cls(task_id=task_id, **config_overrides)
        
        # Load test_metadata.json
        metadata_path = task_dir / "test_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load from metadata
        task.language = metadata.get("language", "python")
        task.test_framework = metadata.get("test_framework", "pytest")
        task.test_command = metadata.get("test_command", "")
        task.test_files = metadata.get("test_files", [])
        task.fail_to_pass = metadata.get("FAIL_TO_PASS", [])
        task.pass_to_pass = metadata.get("PASS_TO_PASS", [])
        
        # Load text files
        def read_file(filename: str, default: str = "") -> str:
            path = task_dir / filename
            return path.read_text() if path.exists() else default
        
        task.problem_statement = read_file("problem_statement.md")
        task.prompt_statement = read_file("prompt_statement.md", task.problem_statement)
        task.test_patch = read_file("test.patch")
        task.golden_patch = read_file("golden.patch")
        task.interface = read_file("interface.md")
        
        # Load requirements
        req_path = task_dir / "requirements.json"
        if req_path.exists():
            with open(req_path) as f:
                task.requirements = json.load(f)
        
        task.image_uri = task.get_default_image_uri()
        return task
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def get_default_image_uri(self) -> str:
        """Get Docker image URI for this task."""
        if hasattr(self, 'config') and self.config:
            return self.config.get_image_uri(self.task_id)
        return f"swe-bench-ext-{self.task_id}:latest"
    
    def get_image_uri(self) -> str:
        """Alias for get_default_image_uri."""
        return self.get_default_image_uri()
    
    def get_golden_solution(self) -> str:
        """Get the golden patch for this task."""
        return self.golden_patch
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for the agent."""
        return f"""You are an expert software engineer tasked with fixing a bug in a codebase.

LANGUAGE: {self.language}
TEST FRAMEWORK: {self.test_framework}
WORKING DIRECTORY: {self.workdir}

INSTRUCTIONS:
1. Analyze the problem statement carefully
2. Explore the codebase to understand the issue
3. Make minimal, focused changes to fix the bug
4. Do NOT modify test files unless explicitly required
5. Ensure your changes don't break existing functionality

When you're done, your solution will be tested automatically against the test suite.
Make sure your changes are complete and the code compiles/runs correctly."""
    
    def get_initial_user_prompt(
        self,
        include_interface: bool = True,
        include_requirements: bool = True,
        include_hints: bool = False,
    ) -> str:
        """
        Generate initial user message with the problem.
        
        Args:
            include_interface: Include interface documentation
            include_requirements: Include requirements list
            include_hints: Include hints (if available)
        """
        problem_text = self.prompt_statement or self.problem_statement
        
        prompt_parts = [
            "## Problem Statement",
            problem_text,
        ]
        
        if include_interface and self.interface:
            prompt_parts.extend([
                "",
                "## Interface",
                self.interface,
            ])
        
        if include_requirements and self.requirements:
            prompt_parts.extend([
                "",
                "## Requirements",
            ])
            for req in self.requirements:
                prompt_parts.append(f"- {req}")
        
        if self.test_files:
            prompt_parts.extend([
                "",
                "## Test Files (for reference)",
                "The following test files will be used to verify your solution:",
            ])
            for tf in self.test_files:
                prompt_parts.append(f"- {tf}")
        
        return "\n".join(prompt_parts)
    
    def generate_setup_script(self) -> Union[str, List[str]]:
        """Generate setup scripts (run before agent starts)."""
        return [
            f"cd {self.workdir}",
            "git status",
            "git log --oneline -3",
        ]
    
    def generate_grading_setup_script(self) -> Union[str, List[str]]:
        """Generate grading setup scripts (apply test patch)."""
        if not self.test_patch:
            return [f"cd {self.workdir}", "echo 'No test patch to apply'"]
        
        # Encode test patch to avoid shell escaping issues
        encoded_patch = base64.b64encode(self.test_patch.encode()).decode()
        
        return [
            f"cd {self.workdir}",
            f"echo '{encoded_patch}' | base64 -d | git apply -v --allow-empty || echo 'Patch may already be applied'",
        ]
    
    def generate_test_run_script(self) -> Union[str, List[str]]:
        """Generate test execution script."""
        if self.test_command:
            return f"cd {self.workdir} && {self.test_command}"
        
        # Default commands per test framework
        default_commands = {
            "pytest": "pytest -xvs",
            "go": "go test -v ./...",
            "jest": "npm test",
            "vitest": "npm test",
            "cargo": "cargo test",
            "maven": "mvn test",
            "gradle": "gradle test",
        }
        
        cmd = default_commands.get(self.test_framework, "echo 'No test command specified'")
        return f"cd {self.workdir} && {cmd}"
    
    def parse_test_results(self, test_output: str) -> "TestSummary":
        """Parse test output into structured results."""
        if EVAL_FRAMEWORK_AVAILABLE:
            return create_test_summary_from_output(
                test_output=test_output,
                framework=self.test_framework,
                fail_to_pass=self.fail_to_pass,
                pass_to_pass=self.pass_to_pass,
            )
        else:
            # Fallback for standalone testing
            return self._fallback_parse_results(test_output)
    
    def _fallback_parse_results(self, test_output: str) -> Dict[str, Any]:
        """Fallback parser when eval-framework not available."""
        output_lower = test_output.lower()
        
        has_failures = any(
            indicator in output_lower
            for indicator in ["fail", "error", "failed", "errors"]
        )
        
        has_passes = any(
            indicator in output_lower
            for indicator in ["pass", "passed", "ok", "success"]
        )
        
        if has_failures:
            score = 0.0
        elif has_passes:
            score = 1.0
        else:
            score = 0.0
        
        return {
            "score": score,
            "test_statuses": {},
            "raw_output": test_output[:1000],
        }
    
    # =========================================================================
    # Additional Methods (eval_service compatible)
    # =========================================================================
    
    def generate_final_solution_state_cmd(self, solution: str) -> str:
        """Generate command to apply solution patch (eval_service compatible)."""
        encoded_solution = base64.b64encode(solution.encode()).decode()
        return f"cd {self.workdir} && echo '{encoded_solution}' | base64 -d | git apply -v"
    
    def generate_apply_solution_script(self, solution: str) -> str:
        """Alias for generate_final_solution_state_cmd."""
        return self.generate_final_solution_state_cmd(solution)
    
    def generate_solution_fetch_script(self) -> str:
        """Generate script to capture agent's solution (eval_service compatible)."""
        return f"cd {self.workdir} && git diff HEAD"
    
    def generate_capture_solution_script(self) -> str:
        """Alias for generate_solution_fetch_script."""
        return self.generate_solution_fetch_script()
    
    def get_sandbox_config(self) -> Dict[str, Any]:
        """Get sandbox configuration for this task."""
        return {
            "image": self.get_default_image_uri(),
            "workdir": self.workdir,
            "timeout": 3600,
        }
    
    # =========================================================================
    # Metadata & Utilities
    # =========================================================================
    
    @property
    def workdir(self) -> str:
        """Working directory inside the sandbox."""
        if hasattr(self, 'config') and self.config and self.config.workdir:
            return self.config.workdir
        return "/workspace/repo"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get task metadata for logging/storage."""
        return {
            "task_id": self.task_id,
            "language": self.language,
            "test_framework": self.test_framework,
            "num_fail_to_pass": len(self.fail_to_pass) if self.fail_to_pass else 0,
            "num_pass_to_pass": len(self.pass_to_pass) if self.pass_to_pass else 0,
            "num_test_files": len(self.test_files) if self.test_files else 0,
            "has_golden_patch": bool(self.golden_patch),
            "has_interface": bool(self.interface),
            "has_requirements": bool(self.requirements),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "task_id": self.task_id,
            "language": self.language,
            "test_framework": self.test_framework,
            "test_command": self.test_command,
            "test_files": self.test_files,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "problem_statement": self.problem_statement,
            "prompt_statement": self.prompt_statement,
            "test_patch": self.test_patch,
            "golden_patch": self.golden_patch,
            "requirements": self.requirements,
            "interface": self.interface,
            "workdir": self.workdir,
            "image_uri": getattr(self, 'image_uri', None),
        }
    
    def __repr__(self) -> str:
        return (
            f"SweBenchExtTask("
            f"task_id='{self.task_id}', "
            f"language='{self.language}', "
            f"test_framework='{self.test_framework}'"
            f")"
        )
