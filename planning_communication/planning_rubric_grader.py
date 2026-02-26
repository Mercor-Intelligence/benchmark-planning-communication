"""
Planning rubric grader — evaluates generated plans (not code).

Implements weighting (major=2, minor=1), adapted for lighthouse's
BaseRubricGrader interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lighthouse.core.grading.rubric.models import Rubric, RubricEvalCriteria
from swe_bench_ext.rubric_grader import SweBenchExtRubricGrader


class PlanningRubricGrader(SweBenchExtRubricGrader):
    """
    Grades generated plans against ``rubric/planning.json`` criteria.

    Scoring uses the imperium-tooling weighting: major=2, minor=1.
    Normalized score = earned_weight / total_weight.
    """

    # -- Rubric parsing --------------------------------------------------------

    def _get_rubric_from_dict(self, rubric_dict: Dict[str, Any]) -> Rubric:
        """
        Parse P&C planning rubric (category-keyed JSON) into a lighthouse Rubric.

        Expected input::

            {
                "functional": [
                    {"id": "functional-1", "description": "...", "weight": "major"},
                    ...
                ],
                "robustness": [...],
                "style": [...]
            }
        """
        criteria: list[RubricEvalCriteria] = []
        categories: Dict[str, list[str]] = {}

        # Process in deterministic order matching imperium convention
        ordered_cats = ["functional", "robustness", "style"]
        seen: set[str] = set()
        for cat in ordered_cats + sorted(rubric_dict.keys()):
            if cat in seen or cat not in rubric_dict:
                continue
            seen.add(cat)
            items = rubric_dict[cat]
            if not isinstance(items, list):
                continue

            cat_ids: list[str] = []
            for item in items:
                cid = item.get("id") or item.get("criteria_id", "")
                criteria.append(
                    RubricEvalCriteria(
                        criteria_id=cid,
                        description=item.get("description", ""),
                        weight=item.get("weight", "major"),
                    )
                )
                cat_ids.append(cid)
            if cat_ids:
                categories[cat] = cat_ids

        return Rubric(criteria=criteria, categories=categories or None, ctx_cmds=[])

    # -- Prompt ----------------------------------------------------------------

    def _build_prompt(
        self,
        trajectory: str,
        rubric: Rubric,
        git_diff: str = "",
        problem_statement: str = "",
        repo_context: str = "",
        grading_guidelines: str = "",
    ) -> str:
        """
        Build a plan-evaluation prompt.

        Adapted from imperium-tooling ``grade_planning.txt`` — evaluates a
        *plan document*, not code.
        """
        rubric_text = self._format_rubric_for_prompt(rubric)

        return f"""# Planning Validation Prompt

You are an expert code reviewer evaluating a software engineering plan.

You will receive:

1. A planning document — an implementation plan for a software task
2. A rubric with criteria organized by category (functional, robustness, style, etc.)

## Your Task

For each criterion in the rubric, determine if the plan adequately addresses it.

A criterion is "met" if the plan:
-   Explicitly describes implementing/handling the requirement, OR
-   Clearly implies the requirement will be addressed through described approach, OR
-   The requirement is implicitly covered by the overall approach

A criterion is "not met" if:
-   The plan does not mention or address the requirement
-   The plan explicitly excludes or deprioritizes the requirement
-   The described approach would likely not satisfy the requirement

## Evaluation Guidelines

-   **Be strict but fair**: Plans don't need exact wording matches, but must demonstrate clear intent to address each criterion
-   **Consider implicit coverage**: A well-structured plan may address multiple criteria through a single comprehensive approach
-   **Functional criteria**: Focus on whether the plan describes implementing the core functionality
-   **Robustness criteria**: Check if the plan considers error handling, edge cases, and defensive programming
-   **Style criteria**: Evaluate if the plan mentions code organization, naming, documentation, or follows project conventions

{f"## Additional Guidelines{chr(10)}{grading_guidelines}" if grading_guidelines else ""}

## Response Format

Return a JSON object with this exact structure:
```json
{{
  "verdicts": [
    {{
      "criterion_id": "functional-1",
      "passed": true,
      "reason": "1-2 sentences explaining your decision"
    }}
  ]
}}
```

Be thorough and evaluate EVERY criterion provided in the rubric.

---

## Planning Statement

{problem_statement}

---

## Plan

{trajectory}

---

## Rubric

{rubric_text}"""

    # -- Scoring (imperium weighting: major=2, minor=1) ------------------------

    def _calculate_total_score(
        self,
        verdicts: Dict[str, Dict[str, Any]],
        rubric: Rubric,
        category_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """Weighted score: major criteria count 2x, minor count 1x."""
        total_weight = 0
        earned_weight = 0
        for c in rubric.criteria:
            w = 2 if c.weight in ("major", 1.0) else 1
            total_weight += w
            if verdicts.get(c.criteria_id, {}).get("passed", False):
                earned_weight += w
        return round(earned_weight / total_weight, 4) if total_weight > 0 else 0.0

    def _verdicts_to_criteria_scores(self, verdicts, rubric):
        """Match scoring weights with _calculate_total_score."""
        from lighthouse.core.grading.rubric.models import RubricCriteriaScore

        scores = []
        for c in rubric.criteria:
            verdict = verdicts.get(c.criteria_id, {})
            passed = verdict.get("passed", False)
            w = 2.0 if c.weight in ("major", 1.0) else 1.0
            scores.append(
                RubricCriteriaScore(
                    criteria=c,
                    score=w if passed else 0.0,
                    explanation=verdict.get("reason", ""),
                )
            )
        return scores
