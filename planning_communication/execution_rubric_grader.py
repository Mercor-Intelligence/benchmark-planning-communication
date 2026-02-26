"""
Execution rubric grader — evaluates code solutions against P&C execution rubrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lighthouse.core.grading.rubric.models import Rubric, RubricEvalCriteria
from swe_bench_ext.rubric_grader import SweBenchExtRubricGrader


class ExecutionRubricGrader(SweBenchExtRubricGrader):
    """
    Grades code solutions against ``rubric/execution.json``.

    Uses the same major=2/minor=1 weighting as PlanningRubricGrader
    for consistency with the imperium-tooling scoring system.
    """

    # P&C execution rubrics follow this category order
    CATEGORY_ORDER = ("correctness", "functional", "robustness", "style")

    # -- Rubric parsing --------------------------------------------------------

    def _get_rubric_from_dict(self, rubric_dict: Dict[str, Any]) -> Rubric:
        """
        Parse P&C execution rubric (category-keyed JSON) into a lighthouse Rubric.

        Processes categories in deterministic order so criteria IDs are stable.
        """
        criteria: list[RubricEvalCriteria] = []
        categories: Dict[str, list[str]] = {}

        seen: set[str] = set()
        for cat in list(self.CATEGORY_ORDER) + sorted(rubric_dict.keys()):
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
        Build an execution-evaluation prompt.

        Adapted from imperium-tooling ``grade_execution.txt``.
        """
        rubric_text = self._format_rubric_for_prompt(rubric)

        return f"""# Execution Validation Prompt

You are an expert code reviewer evaluating a code implementation (patch/diff).

You will receive:
1. A problem statement describing what needed to be built
2. A code patch showing the implementation
3. A rubric with criteria organized by category (correctness, functional, robustness, style)

## Your Task

For each criterion in the rubric, determine if the implementation satisfies it based on the code changes.

A criterion is "met" if:
- The code clearly implements the required functionality
- The changes demonstrate the described behavior
- For test-related correctness criteria: assume tests pass if the relevant code changes are present

A criterion is "not met" if:
- The code does not address the requirement
- The implementation appears incomplete or incorrect
- The approach would likely not satisfy the requirement

## Evaluation Guidelines

- **Correctness criteria**: Focus on whether the code would pass the specified tests. If the patch includes relevant test-passing code changes, assume correctness unless clearly broken
- **Functional criteria**: Verify the code implements the described functionality. Look for the specific features, methods, or behaviors
- **Robustness criteria**: Check for error handling, input validation, edge case handling, and defensive programming
- **Style criteria**: Evaluate code organization, naming conventions, documentation, and adherence to project patterns

### Special Notes

- **Be strict on functional requirements**: Core functionality must be clearly present
- **Be more lenient on style**: Unless explicitly violated or egregiously poor
- **Focus on what's present**: Evaluate based on the actual code changes, not hypotheticals

{f"## Additional Guidelines{chr(10)}{grading_guidelines}" if grading_guidelines else ""}

## Response Format

Return a JSON object with this exact structure:
```json
{{{{
  "verdicts": [
    {{{{
      "criterion_id": "functional-1",
      "passed": true,
      "reason": "1-2 sentences explaining your decision based on specific code evidence"
    }}}}
  ]
}}}}
```

Be thorough and evaluate EVERY criterion provided in the rubric.

---

## Problem Statement

{problem_statement}

---

## Code Changes (Git Diff)

```diff
{git_diff}
```

---

{f"## Agent Work History{chr(10)}{trajectory}" if trajectory and trajectory != git_diff else ""}

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
