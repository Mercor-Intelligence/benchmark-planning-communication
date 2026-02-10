#!/usr/bin/env python3
"""
Offline rubric re-grading script.

Re-runs the SweBenchExtRubricGrader on existing log data (trajectory, patch,
rubric criteria from test_results.json) without spinning up sandboxes or new
evaluation runs.

Usage:
    # Dry run on 5 tasks for a single model
    python scripts/regrade_rubric.py \
        --logs-dir logs/novice_no_reminder_no_exclusions \
        --model anthropic_claude-opus-4-5-20251101 \
        --rubric-model openai/gpt-4o \
        --limit 5 \
        --dry-run

    # Full re-grade for all models
    python scripts/regrade_rubric.py \
        --logs-dir logs/novice_no_reminder_no_exclusions \
        --rubric-model anthropic/claude-sonnet-4-20250514 \
        --concurrency 10
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds; actual delay = base * 2^attempt + jitter

# Add project root to path so we can import swe_bench_ext
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from swe_bench_ext.rubric_grader import SweBenchExtRubricGrader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def discover_runs(
    logs_dir: Path,
    model_filter: Optional[str] = None,
    task_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Walk the logs directory and collect all valid run triples.

    Returns a list of dicts with keys:
        model, task, run, test_results_path, trajectory_path, patch_path
    """
    runs: List[Dict[str, Any]] = []

    # Level 1: model directories
    model_dirs = sorted(logs_dir.iterdir()) if logs_dir.is_dir() else []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_filter and model_filter != model_name:
            continue

        # Level 2: task directories
        task_dirs = sorted(model_dir.iterdir())
        for task_dir in task_dirs:
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if task_filter and task_filter not in task_name:
                continue

            # Level 3: run directories
            run_dirs = sorted(task_dir.iterdir())
            for run_dir in run_dirs:
                if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
                    continue

                test_results_path = run_dir / "test_results.json"
                trajectory_path = run_dir / "trajectory.json"
                patch_path = run_dir / "agent.patch"

                if not test_results_path.exists():
                    continue

                runs.append({
                    "model": model_name,
                    "task": task_name,
                    "run": run_dir.name,
                    "test_results_path": test_results_path,
                    "trajectory_path": trajectory_path,
                    "patch_path": patch_path,
                })

                if limit is not None and len(runs) >= limit:
                    return runs

    return runs


def extract_rubric_dict_from_test_results(test_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Reconstruct a harness-format rubric dict from the existing
    rubric_grade_summary.criteria_scores in test_results.json.

    The harness format expected by convert_harness_rubric_to_framework() is:
        {
            "functional": [{"id": "functional-1", "description": "...", "weight": "major"}],
            "robustness": [...],
            ...
        }

    We reverse-engineer this by grouping criteria by their ID prefix.
    """
    rubric_summary = test_results.get("rubric_grade_summary")
    if not rubric_summary:
        return None

    criteria_scores = rubric_summary.get("criteria_scores", [])
    if not criteria_scores:
        return None

    rubric_dict: Dict[str, list] = {}

    for cs in criteria_scores:
        criteria = cs.get("criteria", {})
        criteria_id = criteria.get("criteria_id", "")
        description = criteria.get("description", "")
        weight = criteria.get("weight", "major")

        if not criteria_id or not description:
            continue

        # Derive category from criteria_id: "functional-1" -> "functional"
        # Handle multi-word categories like "correctness-1"
        parts = criteria_id.rsplit("-", 1)
        category = parts[0] if len(parts) == 2 and parts[1].isdigit() else "functional"

        rubric_dict.setdefault(category, []).append({
            "id": criteria_id,
            "description": description,
            "weight": weight,
        })

    return rubric_dict if rubric_dict else None


def format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    """
    Format trajectory messages into a readable string for the rubric grader.

    Mirrors the approach in InspectAdapter._format_trajectory_for_rubric().
    """
    if not trajectory:
        return ""

    lines = []
    for msg in trajectory:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "") or ""

        lines.append(f"[{role}]")
        if content:
            lines.append(content)

        # Format tool calls if present
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func_name = "unknown"
                if isinstance(tc, dict):
                    func_name = tc.get("function", tc.get("name", "unknown"))
                    if isinstance(func_name, dict):
                        func_name = func_name.get("name", "unknown")
                lines.append(f"  Tool: {func_name}")

        lines.append("")  # blank line between messages

    return "\n".join(lines)


def extract_problem_statement(trajectory: List[Dict[str, Any]]) -> str:
    """
    Extract the problem statement from the trajectory.

    The user message containing "Please solve the following coding issue:" has
    the full problem description + requirements.
    """
    for msg in trajectory:
        content = msg.get("content", "") or ""
        if "Please solve the following coding issue:" in content:
            return content
    return ""


def serialize_rubric_grade_summary(summary) -> Dict[str, Any]:
    """
    Serialize a RubricGradeSummary to match the existing JSON format in
    test_results.json.

    The existing format uses:
    - "criteria_scores": list of {criteria: {...}, score, explanation}
    - "explanation": string
    - "total_weighted_score": float (NOT "total_score")
    """
    criteria_scores_list = []
    for cs in summary.criteria_scores:
        criteria_scores_list.append({
            "criteria": {
                "criteria_id": cs.criteria.criteria_id,
                "description": cs.criteria.description,
                "weight": cs.criteria.weight,
            },
            "score": cs.score,
            "explanation": cs.explanation,
        })

    # Compute normalized score (0-1)
    max_possible = summary.max_possible_score
    normalized = summary.total_score / max_possible if max_possible > 0 else 0.0

    return {
        "criteria_scores": criteria_scores_list,
        "explanation": summary.explanation,
        "total_weighted_score": round(normalized, 4),
    }


def infer_api_key(rubric_model: str) -> Optional[str]:
    """Infer the API key from environment variables based on model provider."""
    provider = rubric_model.split("/")[0].lower() if "/" in rubric_model else ""
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    elif provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    elif provider in ("google", "gemini"):
        return (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
        )
    return None


# ---------------------------------------------------------------------------
# Core grading logic
# ---------------------------------------------------------------------------

async def grade_single_run(
    run_info: Dict[str, Any],
    rubric_model: str,
    api_key: Optional[str],
    dry_run: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """
    Grade a single run and write results in place.

    Returns a result dict with status info.
    """
    label = f"{run_info['model']}/{run_info['task']}/{run_info['run']}"

    try:
        # 1. Read test_results.json
        test_results_path: Path = run_info["test_results_path"]
        with open(test_results_path, "r") as f:
            test_results = json.load(f)

        # 2. Extract rubric dict from existing criteria
        rubric_dict = extract_rubric_dict_from_test_results(test_results)
        if not rubric_dict:
            return {"label": label, "status": "skipped", "reason": "no rubric criteria found"}

        total_criteria = sum(len(v) for v in rubric_dict.values())

        # 3. Read trajectory
        trajectory_path: Path = run_info["trajectory_path"]
        trajectory = []
        if trajectory_path.exists():
            with open(trajectory_path, "r") as f:
                trajectory = json.load(f)

        # 4. Read patch
        patch_path: Path = run_info["patch_path"]
        patch = ""
        if patch_path.exists():
            with open(patch_path, "r") as f:
                patch = f.read()

        # 5. Format inputs
        trajectory_str = format_trajectory(trajectory)
        problem_statement = extract_problem_statement(trajectory)

        if dry_run:
            return {
                "label": label,
                "status": "dry_run",
                "criteria_count": total_criteria,
                "categories": list(rubric_dict.keys()),
                "has_trajectory": bool(trajectory),
                "has_patch": bool(patch),
                "has_problem_statement": bool(problem_statement),
                "trajectory_length": len(trajectory_str),
                "patch_length": len(patch),
            }

        # 6. Grade
        if semaphore:
            async with semaphore:
                return await _do_grade(
                    label, test_results, test_results_path,
                    rubric_dict, patch, trajectory_str, problem_statement,
                    rubric_model, api_key,
                )
        else:
            return await _do_grade(
                label, test_results, test_results_path,
                rubric_dict, patch, trajectory_str, problem_statement,
                rubric_model, api_key,
            )

    except Exception as e:
        logger.error(f"[{label}] Error: {e}")
        return {"label": label, "status": "error", "error": str(e)}


async def _do_grade(
    label: str,
    test_results: Dict[str, Any],
    test_results_path: Path,
    rubric_dict: Dict[str, Any],
    patch: str,
    trajectory_str: str,
    problem_statement: str,
    rubric_model: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    """Execute the actual LLM grading call with retry + backoff, then write results."""
    start = time.time()

    # Create grader
    grader = SweBenchExtRubricGrader(
        model_name=rubric_model,
        api_key=api_key,
    )

    # Load rubric from reconstructed dict
    grader.load_rubric_from_dict(rubric_dict)

    # Retry loop with exponential backoff
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await grader.grade(
                solution=patch,
                trajectory={"transcript": trajectory_str},
                git_diff=patch,
                problem_statement=problem_statement,
            )
            break  # success
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    f"[{label}] Attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{label}] FAILED after {MAX_RETRIES} retries. "
                    f"Last error: {e}"
                )
                raise

    elapsed = time.time() - start

    # Serialize to match existing JSON format
    serialized = serialize_rubric_grade_summary(result)

    # Write back in place
    test_results["rubric_grade_summary"] = serialized
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    return {
        "label": label,
        "status": "graded",
        "total_weighted_score": serialized["total_weighted_score"],
        "elapsed_seconds": round(elapsed, 1),
        "test_results_path": str(test_results_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    """Async entry point."""

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"Error: --logs-dir '{logs_dir}' is not a directory")
        sys.exit(1)

    # Discover runs
    print(f"Discovering runs in {logs_dir} ...")
    runs = discover_runs(
        logs_dir,
        model_filter=args.model,
        task_filter=args.task,
        limit=args.limit,
    )
    print(f"Found {len(runs)} runs to process")

    if not runs:
        print("No runs found. Check --logs-dir, --model, and --task filters.")
        return

    # Infer API key
    api_key = infer_api_key(args.rubric_model)
    if not api_key and not args.dry_run:
        print(f"Warning: Could not infer API key for provider in '{args.rubric_model}'. "
              "Set the appropriate environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process all runs
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Processing {len(runs)} runs "
          f"with rubric model: {args.rubric_model} (concurrency={args.concurrency})\n")

    tasks = [
        grade_single_run(
            run_info=run,
            rubric_model=args.rubric_model,
            api_key=api_key,
            dry_run=args.dry_run,
            semaphore=semaphore,
        )
        for run in runs
    ]

    # Gather with progress
    completed = 0
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        results.append(result)

        status = result.get("status", "unknown")
        label = result.get("label", "?")

        if status == "dry_run":
            criteria_count = result.get("criteria_count", 0)
            has_patch = result.get("has_patch", False)
            has_traj = result.get("has_trajectory", False)
            has_ps = result.get("has_problem_statement", False)
            print(f"  [{completed}/{len(runs)}] {label}: "
                  f"{criteria_count} criteria, "
                  f"patch={'yes' if has_patch else 'no'}, "
                  f"trajectory={'yes' if has_traj else 'no'}, "
                  f"problem_stmt={'yes' if has_ps else 'no'}")
        elif status == "graded":
            score = result.get("total_weighted_score", 0)
            elapsed = result.get("elapsed_seconds", 0)
            print(f"  [{completed}/{len(runs)}] {label}: "
                  f"score={score:.4f} ({elapsed}s)")
        elif status == "skipped":
            reason = result.get("reason", "")
            print(f"  [{completed}/{len(runs)}] {label}: SKIPPED ({reason})")
        elif status == "error":
            error = result.get("error", "")
            print(f"  [{completed}/{len(runs)}] {label}: ERROR: {error}")

    # Summary
    graded = [r for r in results if r["status"] == "graded"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    dry_runs = [r for r in results if r["status"] == "dry_run"]

    print(f"\n--- Summary ---")
    if args.dry_run:
        print(f"  Dry run: {len(dry_runs)} runs discovered")
    else:
        print(f"  Graded: {len(graded)}")
        if graded:
            scores = [r["total_weighted_score"] for r in graded]
            print(f"  Average score: {sum(scores) / len(scores):.4f}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    - {e['label']}: {e['error']}")

    # Write list of graded task paths to file
    graded_tasks_file = Path(args.graded_tasks_file)
    if graded and not args.dry_run:
        graded_tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(graded_tasks_file, "w") as f:
            for r in sorted(graded, key=lambda x: x["label"]):
                f.write(f"{r['test_results_path']}\n")
        print(f"\n  Graded task paths written to: {graded_tasks_file}")
    elif args.dry_run:
        print(f"\n  [DRY RUN] Would write graded task paths to: {graded_tasks_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run rubric grading on existing log data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Path to the top-level logs directory (e.g. logs/novice_no_reminder_no_exclusions)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a single model subdirectory (e.g. anthropic_claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--rubric-model",
        required=True,
        help="LLM model for rubric grading (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of runs to grade (useful for testing / cost control)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Filter to tasks whose name contains this substring",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of parallel LLM calls (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover runs and reconstruct rubrics but skip LLM calls and file writes",
    )
    parser.add_argument(
        "--graded-tasks-file",
        default="mercor_playground/graded_tasks.txt",
        help="Path to write the list of graded test_results.json paths (default: mercor_playground/graded_tasks.txt)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
