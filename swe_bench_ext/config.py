"""
Configuration for SWE-Bench Extended benchmark.

This module defines:
- SweBenchExtConfig: Benchmark-level configuration (extends BenchmarkConfig)
- SweBenchExtOptions: Runtime options for task execution

Author: Mercor Intelligence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field

# Add eval-framework to path if running as submodule
_eval_framework_path = Path(__file__).parent.parent / "eval-framework"
if _eval_framework_path.exists() and str(_eval_framework_path) not in sys.path:
    sys.path.insert(0, str(_eval_framework_path))

# Import from eval-framework (graceful fallback for standalone testing)
try:
    from core.benchmark_tasks.benchmark_config import BenchmarkConfig, BenchmarkType
    EVAL_FRAMEWORK_AVAILABLE = True
except ImportError:
    EVAL_FRAMEWORK_AVAILABLE = False
    BenchmarkConfig = BaseModel

    class BenchmarkType:
        CODE = "code"
        TERMINAL = "terminal"


class SweBenchExtConfig(BenchmarkConfig if EVAL_FRAMEWORK_AVAILABLE else BaseModel):
    """
    Configuration for SWE-Bench-Ext benchmark.
    
    Extends BenchmarkConfig from eval-framework for compatibility.
    Defines where tasks are stored and how to access them.
    """
    
    # Benchmark identity
    name: str = Field(default="swe-bench-ext", description="Benchmark name")
    type: str = Field(default="code", description="Benchmark type (code/terminal)")
    workdir: str = Field(default="/workspace/repo", description="Working directory in sandbox")
    
    # S3 configuration for task data
    s3_bucket: str = Field(
        default="apex-evals-swe-bench",
        description="S3 bucket containing task zip files",
    )
    s3_folder_path: str = Field(
        default="tasks",
        description="Folder path within S3 bucket",
    )
    
    # ECR configuration for Docker images
    ecr_registry: str = Field(
        default="612492817237.dkr.ecr.us-east-1.amazonaws.com",
        description="ECR registry URL",
    )
    ecr_repository: str = Field(
        default="swe-bench-ext",
        description="ECR repository name for task images",
    )
    
    # Image naming
    image_tag_prefix: str = Field(
        default="swe-bench-ext-",
        description="Prefix for Docker image tags",
    )
    
    @property
    def s3_uri(self) -> str:
        """Full S3 URI for task data."""
        return f"s3://{self.s3_bucket}/{self.s3_folder_path}"
    
    def get_image_uri(self, task_id: str) -> str:
        """Get Docker image URI for a specific task."""
        if self.ecr_registry:
            return f"{self.ecr_registry}/{self.ecr_repository}:{task_id}"
        return f"{self.image_tag_prefix}{task_id}:latest"


class SweBenchExtOptions(BaseModel):
    """
    Runtime options for SWE-Bench-Ext task execution.
    
    These can be passed at runtime to customize behavior.
    Compatible with eval-framework's options handling.
    """
    
    # Agent execution options
    message_limit: int = Field(
        default=100,
        ge=1,
        description="Maximum number of agent messages",
    )
    token_limit: Optional[int] = Field(
        default=None,
        description="Maximum tokens for agent (None = unlimited)",
    )
    
    # Task filtering options
    excluded_context: List[str] = Field(
        default_factory=list,
        description="Context to exclude from prompts (e.g., 'hints', 'golden_patch')",
    )
    include_hints: bool = Field(
        default=False,
        description="Whether to include hints in the prompt",
    )
    include_interface: bool = Field(
        default=True,
        description="Whether to include interface documentation in prompt",
    )
    include_requirements: bool = Field(
        default=True,
        description="Whether to include requirements in prompt",
    )
    
    # Grading options
    test_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout for test execution in seconds",
    )
    run_rubric_grading: bool = Field(
        default=False,
        description="Whether to run LLM rubric grading after tests",
    )
    run_trajectory_analysis: bool = Field(
        default=True,
        description="Whether to run post-eval trajectory analysis",
    )
    
    # Golden patch options (for grading mode)
    apply_golden_patch: bool = Field(
        default=False,
        description="Apply golden patch instead of agent solution (for baseline)",
    )
    
    # TaskSource options (passed to TaskSource)
    s3_uri_override: Optional[str] = Field(
        default=None,
        description="Override S3 URI for task data",
    )
