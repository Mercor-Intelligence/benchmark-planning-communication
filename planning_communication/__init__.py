"""
Planning & Communication benchmark task for lighthouse.

Extends benchmark-swe-bench-ext with:
- Dual-stage execution: plan generation → code implementation
- Dual rubric grading: planning rubric + execution rubric
- P&C-specific task instance fields (planning_statement, golden_plan)

Usage::

    lighthouse execute-single \\
        --benchmark planning_communication \\
        --task-id my-task-123 \\
        --model anthropic/claude-sonnet-4-5-20250929 \\
        --task-source-file config/task_source_config.yaml
"""

from .task import PlanningCommTask, PlanningCommTaskInstance
from .config import PlanningCommConfig
from .planning_rubric_grader import PlanningRubricGrader
from .execution_rubric_grader import ExecutionRubricGrader

__all__ = [
    "PlanningCommTask",
    "PlanningCommTaskInstance",
    "PlanningCommConfig",
    "PlanningRubricGrader",
    "ExecutionRubricGrader",
]

__version__ = "0.1.0"
