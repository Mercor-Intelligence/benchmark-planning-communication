"""
Configuration for SWE-Bench Extended benchmark.

This module defines SweBenchExtConfig for benchmark-level configuration.

Author: Mercor Intelligence
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import Field

from lighthouse.core.benchmark_tasks.benchmark_config import BenchmarkConfig


class SweBenchExtConfig(BenchmarkConfig):
    """
    Configuration for SWE-Bench-Ext benchmark.
    
    Extends BenchmarkConfig from eval-framework for compatibility.
    """
    
    # Benchmark identity
    name: str = Field(default="swe-bench-ext", description="Benchmark name")
    type: str = Field(default="code", description="Benchmark type (code/terminal)")
    workdir: str = Field(default="/workspace/repo", description="Working directory in sandbox")
    excluded_context: List[str] = Field(default_factory=list, description="Context to exclude from prompts")
    
    # Optional ECR/registry image template
    # Customers can set this to use pre-built images from their registry
    # Example: "account.dkr.ecr.region.amazonaws.com/repo:{task_id}"
    image_uri_template: Optional[str] = Field(
        default=None,
        description="Template for Docker image URI with {task_id} placeholder",
    )
    
    # Rubric configuration
    rubric_file: str = Field(
        default="rubric/rubric.json",
        description="Path to rubric file within task directory (relative to task root). "
                    "Set to 'rubric/rubric.json' for legacy format, or "
                    "'rubric/execution.json' / 'rubric/planning.json' for split rubrics."
    )
    
    def get_image_uri(self, task_id: str) -> str:
        """Get Docker image URI for a specific task.
        
        If image_uri_template is set, formats it with the task_id.
        Otherwise, returns a simple local image name.
        """
        if self.image_uri_template:
            return self.image_uri_template.format(task_id=task_id)
        return f"swe-bench-ext-{task_id}:latest"
