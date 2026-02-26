"""
Planning & Communication benchmark task.

Extends SweBenchExtTask with:
- Additional task instance fields (planning_statement, golden_plan, dual rubrics)
- Dual rubric grader attachment (execution via lighthouse grade flow, planning via scripts)

Execution harness evals are IDENTICAL to SWE-Bench-Ext — same stages, same
sandbox flow, same test parsing. The only additions are:
  1. Extra fields on the task instance (planning_statement, golden_plan, rubrics)
  2. Execution rubric grader uses the P&C category format
  3. A planning_rubric_grader attribute for use by standalone plan-grading scripts

Plan generation and plan rubric grading are SEPARATE from execution — they
run outside the lighthouse harness via scripts/run_plan_grading.py.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from pydantic import Field

from lighthouse.core.benchmark_tasks.base_benchmark_task import BaseTaskInstance
from lighthouse.core.benchmark_tasks.benchmark_config import BenchmarkConfig
from lighthouse.core.benchmark_tasks.task_source import FolderTaskSource, TaskSource
from lighthouse.core.registry import benchmark_task
from swe_bench_ext.task import SweBenchExtTask, SweBenchExtTaskInstance

from .config import PlanningCommConfig


# ---------------------------------------------------------------------------
# Task instance model
# ---------------------------------------------------------------------------

class PlanningCommTaskInstance(SweBenchExtTaskInstance):
    """SWE-Bench-Ext task instance extended with P&C-specific fields."""

    planning_statement: str = Field(default="", description="Planning statement prompt")
    golden_plan: str = Field(default="", description="Reference plan (golden)")
    planning_rubric: dict = Field(default_factory=dict, description="Planning rubric JSON")
    execution_rubric: dict = Field(default_factory=dict, description="Execution rubric JSON")


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------

@benchmark_task("planning_communication")
class PlanningCommTask(SweBenchExtTask):
    """
    Planning & Communication benchmark task.

    Registered as ``planning_communication`` in the lighthouse plugin registry.

    Execution is the standard SWE-Bench-Ext single "solve" stage — the agent
    receives the problem and implements a fix in the sandbox. All stage logic,
    script generation, test parsing, etc. are inherited unchanged.

    Plan generation and planning rubric grading are decoupled and run
    separately via ``scripts/run_plan_grading.py``.

    Usage::

        lighthouse execute-single \\
            --benchmark planning_communication \\
            --task-id my-task-123 \\
            --model anthropic/claude-sonnet-4-5-20250929 \\
            --task-source-file config/task_source_config.yaml
    """

    config: PlanningCommConfig
    task_instance: PlanningCommTaskInstance

    config_class: ClassVar[Type[BenchmarkConfig]] = PlanningCommConfig
    supported_task_sources: ClassVar[Tuple[Type, ...]] = (FolderTaskSource,)

    # Planning rubric grader — NOT used by lighthouse's grade flow.
    # Available for standalone plan-grading scripts.
    planning_rubric_grader: Optional[Any] = None

    # =========================================================================
    # Task loading
    # =========================================================================

    @classmethod
    def _load_task(cls, task_id: str, task_source: FolderTaskSource) -> PlanningCommTaskInstance:
        """Load base SWE-Bench-Ext fields then overlay P&C-specific files."""
        base = super()._load_task(task_id, task_source)

        def _load_file(name: str, default: str = "") -> str:
            try:
                return task_source.get_task_file_contents(task_id, name)
            except Exception:
                return default

        def _load_json(name: str) -> dict:
            try:
                raw = task_source.get_task_file_contents(task_id, name)
                return json.loads(raw) if raw else {}
            except Exception:
                return {}

        return PlanningCommTaskInstance(
            **base.model_dump(),
            planning_statement=_load_file("planning_statement.md"),
            golden_plan=_load_file("golden_plan.md"),
            planning_rubric=_load_json("rubric/planning.json"),
            execution_rubric=_load_json("rubric/execution.json"),
        )

    # =========================================================================
    # Factory — attach rubric graders
    # =========================================================================

    @classmethod
    def from_id(
        cls,
        task_id: str,
        task_source: FolderTaskSource,
        image_uri_override: Optional[str] = None,
        **config_overrides,
    ) -> "PlanningCommTask":
        task_instance = cls._load_task(task_id, task_source)
        task = cls(task_instance=task_instance, task_source=task_source, **config_overrides)

        if not task_instance.image_uri:
            task_instance.image_uri = (
                image_uri_override if image_uri_override else task.get_default_image_uri()
            )

        # Execution rubric grader — lighthouse's grade flow calls task.rubric_grader
        if task_instance.execution_rubric:
            from .execution_rubric_grader import ExecutionRubricGrader

            task.rubric_grader = ExecutionRubricGrader()

        # Planning rubric grader — used by scripts/run_plan_grading.py, NOT by lighthouse
        if task_instance.planning_rubric:
            from .planning_rubric_grader import PlanningRubricGrader

            task.planning_rubric_grader = PlanningRubricGrader()

        return task

    # =========================================================================
    # Execution inherits everything from SweBenchExtTask:
    #   - get_stages()         → single "solve" stage
    #   - get_system_prompt()  → standard SWE-Bench-Ext agent prompt
    #   - get_initial_user_prompt() → problem_statement / prompt_statement
    #   - generate_setup_script()
    #   - generate_grading_setup_script()
    #   - generate_test_run_script()
    #   - parse_test_results()
    # =========================================================================

    def _get_rubric_dict(self) -> Dict[str, Any]:
        """Return the execution rubric dict for lighthouse's grade flow."""
        if self.task_instance.execution_rubric:
            return self.task_instance.execution_rubric
        return super()._get_rubric_dict()

    def get_planning_rubric_dict(self) -> Dict[str, Any]:
        """Return the planning rubric dict (for plan-grading scripts)."""
        return self.task_instance.planning_rubric
