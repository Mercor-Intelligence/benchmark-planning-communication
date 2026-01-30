"""
Rubric format conversion utilities.

Converts between harness rubric format and eval-framework Rubric format.
"""

from typing import Dict, Any, List
import sys
from pathlib import Path

from lighthouse.core.grading.rubric.models import (
    Rubric,
    RubricEvalCriteria,
)


def convert_harness_rubric_to_framework(harness_dict: Dict[str, Any]) -> Rubric:
    """
    Convert harness rubric format to framework Rubric.
    
    Harness format (categories as top-level keys):
        {
            "functional": [{"id": "f1", "description": "...", "weight": "major"}],
            "robustness": [...],
            "style": [...],
            "trajectory": [...]
        }
    
    Framework format (flat criteria list with categories):
        {
            "criteria": [{"criteria_id": "f1", "description": "...", "weight": "major"}],
            "categories": {"functional": ["f1", ...]}
        }
    
    Args:
        harness_dict: Rubric dictionary from harness tasks
        
    Returns:
        Rubric object in framework format
    """
    criteria = []
    categories = {}
    
    # Process category keys.
    #
    # "correctness" can be extremely large (1000+ auto-generated criteria), which
    # is not practical for LLM grading. However, for benchmarks/tasks where
    # correctness is intentionally small, we do include it.
    correctness = harness_dict.get("correctness") or []
    metadata = harness_dict.get("metadata") or {}
    num_correctness = metadata.get("num_correctness_criteria")
    include_correctness = False
    try:
        if num_correctness is not None:
            include_correctness = int(num_correctness) <= 50
        else:
            include_correctness = len(correctness) <= 50
    except Exception:
        include_correctness = False

    category_order: List[str] = []
    if include_correctness and "correctness" in harness_dict:
        category_order.append("correctness")
    category_order.extend(["functional", "robustness", "style", "trajectory"])

    for category_name in category_order:
        if category_name in harness_dict:
            category_criteria = harness_dict[category_name]
            category_ids = []
            
            for item in category_criteria:
                # Create RubricEvalCriteria with framework format
                criterion = RubricEvalCriteria(
                    criteria_id=item["id"],
                    description=item["description"],
                    weight=item.get("weight", "major"),
                )
                criteria.append(criterion)
                category_ids.append(item["id"])
            
            if category_ids:
                categories[category_name] = category_ids
    
    return Rubric(
        criteria=criteria,
        categories=categories if categories else None,
        ctx_cmds=[],
    )


def convert_framework_rubric_to_harness(framework_rubric: Rubric) -> Dict[str, Any]:
    """
    Convert framework Rubric to harness format.
    
    Used for compatibility when calling harness tools.
    
    Args:
        framework_rubric: Rubric in framework format
        
    Returns:
        Dictionary in harness format
    """
    result = {
        "functional": [],
        "robustness": [],
        "style": [],
        "trajectory": [],
    }
    
    if not framework_rubric.categories:
        # No categories - put everything in functional
        for criterion in framework_rubric.criteria:
            result["functional"].append({
                "id": criterion.criteria_id,
                "description": criterion.description,
                "weight": criterion.weight,
            })
    else:
        # Organize by categories
        for category, criterion_ids in framework_rubric.categories.items():
            if category not in result:
                result[category] = []
            
            for criterion in framework_rubric.criteria:
                if criterion.criteria_id in criterion_ids:
                    result[category].append({
                        "id": criterion.criteria_id,
                        "description": criterion.description,
                        "weight": criterion.weight,
                    })
    
    return result

