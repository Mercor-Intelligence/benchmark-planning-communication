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
    from core.benchmark_tasks.base_benchmark_task import (
        BaseBenchmarkTask,
        BaseTaskInstance,
    )
    from core.benchmark_tasks.models import TestSummary, TestStatus
    from core.benchmark_tasks.task_source import TaskSource, FolderTaskSource
    from core.benchmark_tasks.benchmark_config import BenchmarkConfig, BenchmarkType
    from core.parsing import create_test_summary_from_output
    from core.registry import benchmark_task
    
    EVAL_FRAMEWORK_AVAILABLE = True
except ImportError:
    EVAL_FRAMEWORK_AVAILABLE = False
    BaseBenchmarkTask = object
    BaseTaskInstance = None
    benchmark_task = lambda name: lambda cls: cls

from pydantic import BaseModel, Field
from .config import SweBenchExtConfig


# =============================================================================
# Task Instance Model (extends BaseTaskInstance)
# =============================================================================

class SweBenchExtTaskInstance(BaseModel):
    """
    SWE-Bench-Ext specific task instance data.
    
    Extends the base task instance with SWE-Bench specific fields.
    """
    # Required from BaseTaskInstance
    id: str = Field(description="Task identifier")
    image_uri: str = Field(default="", description="Docker image URI")
    
    # SWE-Bench specific fields
    language: str = Field(default="python", description="Programming language")
    test_framework: str = Field(default="pytest", description="Test framework")
    test_command: str = Field(default="", description="Command to run tests")
    test_files: List[str] = Field(default_factory=list, description="Test file paths")
    fail_to_pass: List[str] = Field(default_factory=list, description="Tests that should pass after fix")
    pass_to_pass: List[str] = Field(default_factory=list, description="Tests that should continue to pass")
    
    # Content fields
    problem_statement: str = Field(default="", description="Problem description")
    prompt_statement: str = Field(default="", description="Customized prompt (optional)")
    test_patch: str = Field(default="", description="Patch to apply for testing")
    golden_patch: str = Field(default="", description="Reference solution")
    requirements: List[str] = Field(default_factory=list, description="Requirements list")
    interface: str = Field(default="", description="Interface documentation")


# =============================================================================
# Task Implementation
# =============================================================================

@benchmark_task("swe_bench_ext")
class SweBenchExtTask(BaseBenchmarkTask if EVAL_FRAMEWORK_AVAILABLE else object):
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
    
    # === Class-level configuration ===
    config_class: ClassVar[Type[BenchmarkConfig]] = SweBenchExtConfig
    supported_task_sources: ClassVar[Tuple[Type, ...]] = (FolderTaskSource,) if EVAL_FRAMEWORK_AVAILABLE else ()
    
    # === Instance attributes ===
    task_instance: SweBenchExtTaskInstance
    
    # =========================================================================
    # Task Loading (implements abstract method)
    # =========================================================================
    
    @classmethod
    def _load_task(cls, task_id: str, task_source: TaskSource) -> SweBenchExtTaskInstance:
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
            test_framework=metadata.get("test_framework", "pytest"),
            test_command=metadata.get("test_command", ""),
            test_files=metadata.get("test_files", []),
            fail_to_pass=metadata.get("FAIL_TO_PASS", metadata.get("fail_to_pass", [])),
            pass_to_pass=metadata.get("PASS_TO_PASS", metadata.get("pass_to_pass", [])),
            problem_statement=problem_statement,
            prompt_statement=load_file("prompt_statement.md", problem_statement),
            test_patch=load_file("test.patch"),
            golden_patch=load_file("golden.patch"),
            requirements=requirements,
            interface=load_file("interface.md"),
        )
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def get_default_image_uri(self) -> str:
        """Get Docker image URI for this task."""
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
        
        # Append tool prompts if provided
        if tool_prompts:
            prompt += "\n\n" + "\n".join(tool_prompts)
        
        return prompt
    
    def get_initial_user_prompt(self, tool_prompts: Optional[List[str]] = None) -> str:
        """Generate initial user message with the problem."""
        inst = self.task_instance
        problem_text = inst.prompt_statement or inst.problem_statement
        
        prompt_parts = [
            "## Problem Statement",
            problem_text,
        ]
        
        if inst.interface:
            prompt_parts.extend(["", "## Interface", inst.interface])
        
        if inst.requirements:
            prompt_parts.extend(["", "## Requirements"])
            for req in inst.requirements:
                prompt_parts.append(f"- {req}")
        
        if inst.test_files:
            prompt_parts.extend([
                "",
                "## Test Files (for reference)",
                "The following test files will be used to verify your solution:",
            ])
            for tf in inst.test_files:
                prompt_parts.append(f"- {tf}")
        
        prompt = "\n".join(prompt_parts)
        
        # Append tool prompts if provided
        if tool_prompts:
            prompt += "\n\n" + "\n".join(tool_prompts)
        
        return prompt
    
    def generate_setup_script(self) -> Union[str, List[str]]:
        """Generate setup scripts (run before agent starts)."""
        return [
            f"cd {self.workdir}",
            "git status",
            "git log --oneline -3",
        ]
    
    def generate_grading_setup_script(self) -> Union[str, List[str]]:
        """Generate grading setup scripts (apply test patch)."""
        inst = self.task_instance
        
        if not inst.test_patch:
            return [f"cd {self.workdir}", "echo 'No test patch to apply'"]
        
        # Encode test patch to avoid shell escaping issues
        encoded_patch = base64.b64encode(inst.test_patch.encode()).decode()
        
        return [
            f"cd {self.workdir}",
            f"echo '{encoded_patch}' | base64 -d | git apply -v --allow-empty || echo 'Patch may already be applied'",
        ]
    
    def generate_test_run_script(self) -> Union[str, List[str]]:
        """Generate test execution script."""
        inst = self.task_instance
        
        if inst.test_command:
            return f"cd {self.workdir} && {inst.test_command}"
        
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
        
        cmd = default_commands.get(inst.test_framework, "echo 'No test command specified'")
        return f"cd {self.workdir} && {cmd}"
    
    def parse_test_results(self, test_output: str) -> "TestSummary":
        """Parse test output into structured results."""
        inst = self.task_instance
        
        if EVAL_FRAMEWORK_AVAILABLE:
            return create_test_summary_from_output(
                test_output=test_output,
                framework=inst.test_framework,
                fail_to_pass=inst.fail_to_pass,
                pass_to_pass=inst.pass_to_pass,
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
    # Metadata
    # =========================================================================
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get task metadata for logging/storage."""
        inst = self.task_instance
        return {
            "task_id": inst.id,
            "language": inst.language,
            "test_framework": inst.test_framework,
            "num_fail_to_pass": len(inst.fail_to_pass),
            "num_pass_to_pass": len(inst.pass_to_pass),
            "num_test_files": len(inst.test_files),
            "has_golden_patch": bool(inst.golden_patch),
            "has_interface": bool(inst.interface),
            "has_requirements": bool(inst.requirements),
        }
    
    def __repr__(self) -> str:
        inst = self.task_instance
        return (
            f"SweBenchExtTask("
            f"id='{inst.id}', "
            f"language='{inst.language}', "
            f"test_framework='{inst.test_framework}'"
            f")"
        )
