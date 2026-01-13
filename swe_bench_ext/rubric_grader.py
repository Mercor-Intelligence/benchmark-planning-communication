"""
Concrete rubric grader implementation for SWE-Bench-Ext.

Extends BaseRubricGrader with OpenAI/Anthropic LLM integration.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add eval-framework to path
_eval_framework_path = Path(__file__).parent.parent / "eval-framework"
if _eval_framework_path.exists() and str(_eval_framework_path) not in sys.path:
    sys.path.insert(0, str(_eval_framework_path))

from core.grading.rubric.base_grader import BaseRubricGrader
from core.grading.rubric.models import Rubric

from .rubric_utils import convert_harness_rubric_to_framework


class SweBenchExtRubricGrader(BaseRubricGrader):
    """
    Rubric grader for SWE-Bench-Ext tasks.
    
    Implements BaseRubricGrader abstract methods with OpenAI/Anthropic support.
    
    Usage:
        grader = SweBenchExtRubricGrader(
            rubric=rubric,
            model_name="openai/gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        result = await grader.grade(
            solution=diff,
            git_diff=diff,
            problem_statement=problem,
        )
    """
    
    def _get_rubric_from_dict(self, rubric_dict: Dict[str, Any]) -> Rubric:
        """
        Parse harness rubric format into framework Rubric.
        
        Harness format has categories as top-level keys.
        """
        return convert_harness_rubric_to_framework(rubric_dict)
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using OpenAI or Anthropic.
        
        Supports: openai/*, anthropic/*
        """
        if not self.model_name:
            raise ValueError("model_name not set")
        
        provider = self.model_name.split("/")[0].lower()
        
        if provider == "openai":
            return await self._call_openai(prompt)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        import openai
        
        client = openai.AsyncOpenAI(api_key=self.api_key)
        model = self.model_name.split("/", 1)[1]  # Remove "openai/" prefix
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        model = self.model_name.split("/", 1)[1]  # Remove "anthropic/" prefix
        
        response = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=4096,
        )
        
        return response.content[0].text
    
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
        Build grading prompt for SWE-Bench-Ext tasks.
        
        Emphasizes code changes and test results.
        """
        sections = []
        
        sections.append("You are an expert code reviewer evaluating an AI agent's solution to a software engineering task.")
        
        # Add grading guidelines if provided
        if grading_guidelines:
            sections.append("\n## Grading Guidelines")
            sections.append(grading_guidelines)
        
        # Add rubric
        sections.append("\n## Evaluation Rubric")
        sections.append(self._format_rubric_for_prompt(rubric))
        
        # Add problem context
        if problem_statement:
            sections.append("\n## Problem Statement")
            sections.append(problem_statement)
        
        # Add code changes (MOST IMPORTANT)
        if git_diff:
            sections.append("\n## Agent's Code Changes (Git Diff)")
            sections.append("```diff")
            sections.append(git_diff)
            sections.append("```")
        
        # Add trajectory
        if trajectory:
            sections.append("\n## Agent's Work History")
            sections.append(trajectory[:5000])  # Limit to prevent token overflow
        
        # Instructions
        sections.append("\n## Your Task")
        sections.append("Evaluate each criterion based on the code changes shown above.")
        sections.append("Return a JSON object with this structure:")
        sections.append("""{
  "verdicts": [
    {
      "criterion_id": "functional-1",
      "passed": true,
      "reason": "The agent correctly implemented...",
      "confidence": "high"
    }
  ]
}""")
        
        return "\n".join(sections)
    
    def _calculate_category_scores(
        self,
        verdicts: Dict[str, Dict[str, Any]],
        rubric: Rubric,
    ) -> Dict[str, float]:
        """
        Calculate per-category scores for SWE-Bench-Ext.
        
        Uses category-weighted scoring if rubric has categories.
        """
        if not rubric.categories:
            return {}
        
        category_scores = {}
        
        for category_name, criterion_ids in rubric.categories.items():
            # Get criteria in this category
            category_criteria = [
                c for c in rubric.criteria
                if c.criteria_id in criterion_ids
            ]
            
            if not category_criteria:
                category_scores[category_name] = 0.0
                continue
            
            # Calculate: (earned weight) / (total weight)
            total_weight = sum(c.get_numeric_weight() for c in category_criteria)
            earned_weight = sum(
                c.get_numeric_weight()
                for c in category_criteria
                if verdicts.get(c.criteria_id, {}).get("passed", False)
            )
            
            category_scores[category_name] = (
                earned_weight / total_weight if total_weight > 0 else 0.0
            )
        
        return category_scores

