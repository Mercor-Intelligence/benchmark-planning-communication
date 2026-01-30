"""
Configuration for SWE-Bench Extended benchmark.

This module defines SweBenchExtConfig for benchmark-level configuration.

Author: Mercor Intelligence
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from lighthouse.core.benchmark_tasks.benchmark_config import BenchmarkConfig
from lighthouse.core.benchmark_tasks.models import ArtifactSpec
from lighthouse.core.harness.continuation_policy import ReminderPolicyConfig


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

    # Stage configuration (multi-stage execution)
    enable_pr_artifacts: bool = Field(
        default=False,
        description="Enable a second ARTIFACTS stage that generates PR artifacts",
    )
    artifacts_out_dir: str = Field(
        default="/workspace/repo/.agent_artifacts",
        description="Output directory for generated artifacts (inside sandbox)",
    )
    # Note: artifacts_user_prompt is now hardcoded in task.py (not configurable)
    artifacts_specs: Optional[List[ArtifactSpec]] = Field(
        default=None,
        description=(
            "Optional override for default PR artifact specs. "
            "If set, these specs will be passed to eval-framework's artifacts stage validator."
        ),
    )

    # Generic per-stage configuration knobs (apply to any stage id)
    stage_tools: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Optional per-stage tool overrides (e.g. {solve: [bash], artifacts: [bash]})",
    )
    stage_system_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-stage system prompt overrides keyed by stage id",
    )
    stage_user_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-stage user prompt overrides keyed by stage id",
    )

    # Reminder configuration (uses eval-framework abstraction)
    # Phase 1 implementation: readme_mode and ask_question_mode only
    # Modes: "off" | "constant" (staged/paced modes not yet implemented)
    # Note: artifact_mode exists in ReminderPolicyConfig but is not implemented yet
    reminder_policy: ReminderPolicyConfig = Field(
        default_factory=lambda: ReminderPolicyConfig(
            readme_mode="off",
            ask_question_mode="off",
            artifact_mode="off",  # Not implemented - included for future use
        ),
        description=(
            "Reminder policy configuration using eval-framework's ReminderPolicyConfig. "
            "Phase 1 implements readme_mode and ask_question_mode only. "
            "Supports 'off' and 'constant' modes."
        ),
    )
    
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
