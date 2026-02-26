"""
Planning & Communication benchmark configuration.

Extends SweBenchExtConfig with P&C-specific options for dual rubric files
and QC scoring thresholds.
"""

from __future__ import annotations

from pydantic import Field
from swe_bench_ext.config import SweBenchExtConfig


class PlanningCommConfig(SweBenchExtConfig):
    """
    Configuration for the Planning & Communication benchmark.

    Inherits all SweBenchExtConfig fields (stages, reminders, image URI,
    excluded_context, etc.).

    Execution harness evals use the standard SWE-Bench-Ext flow unchanged.
    Plan generation and planning rubric grading are separate (run via scripts).
    """

    name: str = Field(default="planning-communication", description="Benchmark name")

    # -- Rubric paths (within task directory) ------------------------------------
    planning_rubric_file: str = Field(
        default="rubric/planning.json",
        description="Path to planning rubric within task directory",
    )
    execution_rubric_file: str = Field(
        default="rubric/execution.json",
        description="Path to execution rubric within task directory",
    )

    # -- QC scoring --------------------------------------------------------------
    qc_passing_threshold: float = Field(
        default=0.70,
        description="Minimum QC score to pass (0.0-1.0)",
    )
