#!/usr/bin/env python3
"""
Multi-model plan generation and rubric grading for P&C tasks.

Outputs eval results in the bytedance-pc-pilot format:

    evals/<task-id>/
    ├── _metadata.json
    ├── plan/
    │   ├── golden.json
    │   ├── claude-sonnet-4.5.json
    │   └── ...
    └── execution/
        ├── golden.json
        ├── claude-sonnet-4.5.json
        └── ...

Features:
- Runs N models × M tasks concurrently with adaptive rate limiting
- Exponential backoff with jitter on transient failures (rate limits, timeouts, 5xx)
- Deterministic verification pass after completion — retries all failures
- Graceful resume: skips tasks/models that already have results on disk
- No Modal / sandbox — pure LLM API calls via litellm

Usage:
    # Batch via jsonl (same format as lighthouse --tasks-file)
    python scripts/run_plan_grading.py \\
        --tasks-dir /path/to/tasks \\
        --output-dir evals/ \\
        --plan-models "anthropic/claude-sonnet-4-5-20250929,google/gemini-2.5-pro" \\
        --grade-model openai/gpt-4o \\
        --execution \\
        --workers 30

    # Single task test
    python scripts/run_plan_grading.py \\
        --tasks-dir /path/to/tasks \\
        --task-id my-task-123 \\
        --plan-models anthropic/claude-sonnet-4-5-20250929 \\
        --grade-model openai/gpt-4o

    # tasks.jsonl format (one JSON object per line):
    #   {"task_id": "my-task-123"}
    #   {"task_id": "another-task-456"}
"""

from __future__ import annotations

import warnings

# Suppress Modal warning from lighthouse (this script does not use Modal/AWS)
warnings.filterwarnings("ignore", message=".*blocking Modal.*")
warnings.filterwarnings("ignore", message=".*Modal.*async.*")

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce lighthouse logs so "Using pre-stored Modal secret for AWS" does not appear
logging.getLogger("lighthouse").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def normalize_litellm_model(name: str) -> str:
    """Use LiteLLM provider prefix: e.g. google/gemini-* -> gemini/gemini-*."""
    s = name.strip()
    if s.startswith("google/gemini"):
        return "gemini/" + s.split("/", 1)[1]
    return s


# =============================================================================
# Retry infrastructure
# =============================================================================

TRANSIENT_ERRORS = (
    "rate_limit",
    "timeout",
    "overloaded",
    "server_error",
    "connection",
    "502",
    "503",
    "529",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
)

MAX_RETRIES = 8
BASE_DELAY = 2.0
MAX_DELAY = 120.0


def is_transient(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(t.lower() in msg for t in TRANSIENT_ERRORS)


async def retry_with_backoff(coro_factory, desc: str = ""):
    """
    Call coro_factory() repeatedly with exponential backoff + jitter.

    coro_factory must be a callable that returns a new coroutine each time.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except Exception as e:
            if attempt == MAX_RETRIES or not is_transient(e):
                raise
            delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter
            log.warning(
                f"[{desc}] Attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e} "
                f"— retrying in {wait:.1f}s"
            )
            await asyncio.sleep(wait)


# =============================================================================
# LLM calls
# =============================================================================

PLAN_GENERATION_PROMPT = """You are an expert software engineer. Based on the problem statement and planning instructions below, create a detailed implementation plan.

## Problem Statement

{problem_statement}

## Planning Instructions

{planning_statement}

---

Please provide a comprehensive implementation plan following the instructions above. Be specific about:
- Files to modify or create
- Code changes needed
- The order of implementation steps
- Any dependencies between steps

Format your plan with clear sections and numbered steps."""


async def generate_plan(
    problem_statement: str,
    planning_statement: str,
    model: str,
) -> str:
    """Generate a plan using litellm."""
    import litellm

    prompt = PLAN_GENERATION_PROMPT.format(
        problem_statement=problem_statement,
        planning_statement=planning_statement,
    )

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def grade_against_rubric(
    content: str,
    rubric_dict: dict,
    context: str,
    rubric_type: str,
    model: str,
    api_key: str | None = None,
) -> dict:
    """Grade content against a rubric. Works for both plan and execution grading."""
    if rubric_type == "planning":
        from planning_communication.planning_rubric_grader import PlanningRubricGrader
        grader = PlanningRubricGrader(model_name=model, api_key=api_key)
    else:
        from planning_communication.execution_rubric_grader import ExecutionRubricGrader
        grader = ExecutionRubricGrader(model_name=model, api_key=api_key)

    grader.load_rubric_from_dict(rubric_dict)

    if rubric_type == "planning":
        result = await grader.grade(solution=content, problem_statement=context)
    else:
        result = await grader.grade(
            solution=content, git_diff=content, problem_statement=context,
        )

    earned = sum(cs.score for cs in result.criteria_scores)
    total = sum(
        (2.0 if cs.criteria.weight in ("major", 1.0) else 1.0)
        for cs in result.criteria_scores
    )
    # Use weighted ratio [0, 1]; base-class normalized_score may use a different scale
    score = round(earned / total, 6) if total > 0 else 0.0

    return {
        "score": score,
        "earned_weight": int(earned),
        "total_weight": int(total),
        "criteria_results": [
            {
                "id": cs.criteria.criteria_id,
                "met": cs.score > 0,
                "reasoning": cs.explanation,
            }
            for cs in result.criteria_scores
        ],
        "summary": result.explanation,
    }


# =============================================================================
# Task data loading
# =============================================================================

@dataclass
class TaskData:
    task_id: str
    problem_statement: str = ""
    planning_statement: str = ""
    golden_plan: str = ""
    golden_patch: str = ""
    planning_rubric: dict = field(default_factory=dict)
    execution_rubric: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def load_task(task_id: str, tasks_dir: Path) -> TaskData:
    """Load all task files from disk."""
    d = tasks_dir / task_id

    def _read(name: str) -> str:
        p = d / name
        return p.read_text() if p.exists() else ""

    def _read_json(name: str) -> dict:
        p = d / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    meta = _read_json("test_metadata.json")
    metadata = {
        "task": task_id,
        "language": meta.get("language", ""),
    }
    # Pull category/tag/difficulty from _metadata.json if it exists (bytedance format)
    meta_file = d / "_metadata.json"
    if meta_file.exists():
        try:
            extra = json.loads(meta_file.read_text())
            metadata.update(extra)
        except json.JSONDecodeError:
            pass

    return TaskData(
        task_id=task_id,
        problem_statement=_read("problem_statement.md"),
        planning_statement=_read("planning_statement.md"),
        golden_plan=_read("golden_plan.md"),
        golden_patch=_read("golden.patch"),
        planning_rubric=_read_json("rubric/planning.json"),
        execution_rubric=_read_json("rubric/execution.json"),
        metadata=metadata,
    )


# =============================================================================
# Single unit of work
# =============================================================================

@dataclass
class Job:
    task_id: str
    model: str
    rubric_type: str  # "planning" or "execution"
    is_golden: bool = False


def output_path(output_dir: Path, job: Job) -> Path:
    model_name = "golden" if job.is_golden else job.model
    # Normalize model names for filenames (anthropic/claude-sonnet-4-5 -> claude-sonnet-4-5)
    safe_name = model_name.split("/")[-1] if "/" in model_name else model_name
    return output_dir / job.task_id / job.rubric_type / f"{safe_name}.json"


def is_complete(output_dir: Path, job: Job) -> bool:
    """Check if a job already has results on disk."""
    p = output_path(output_dir, job)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return "score" in data and "criteria_results" in data and "error" not in data
    except (json.JSONDecodeError, KeyError):
        return False


async def run_job(
    job: Job,
    task: TaskData,
    output_dir: Path,
    grade_model: str,
    sem: asyncio.Semaphore,
) -> dict:
    """Execute a single job: generate (if needed) + grade."""
    async with sem:
        out = output_path(output_dir, job)
        out.parent.mkdir(parents=True, exist_ok=True)

        model_label = "golden" if job.is_golden else job.model
        desc = f"{task.task_id}/{job.rubric_type}/{model_label}"
        start = time.time()

        try:
            if job.rubric_type == "planning":
                rubric = task.planning_rubric
                context = task.planning_statement or task.problem_statement

                if job.is_golden:
                    plan_text = task.golden_plan
                else:
                    plan_text = await retry_with_backoff(
                        lambda: generate_plan(
                            task.problem_statement, task.planning_statement, job.model,
                        ),
                        desc=f"{desc}/gen",
                    )

                grade_result = await retry_with_backoff(
                    lambda: grade_against_rubric(
                        plan_text, rubric, context, "planning", grade_model,
                    ),
                    desc=f"{desc}/grade",
                )

            else:  # execution
                rubric = task.execution_rubric
                context = task.problem_statement

                if job.is_golden:
                    patch_text = task.golden_patch
                else:
                    # For non-golden execution grading, we'd need the model's
                    # solution from trajectory.json or a separate execution run.
                    # Skip if no solution available.
                    return {"task": task.task_id, "model": model_label, "status": "skipped",
                            "reason": "execution grading for non-golden requires agent solutions"}

                grade_result = await retry_with_backoff(
                    lambda: grade_against_rubric(
                        patch_text, rubric, context, "execution", grade_model,
                    ),
                    desc=f"{desc}/grade",
                )

            elapsed = time.time() - start

            result = {
                "task": task.task_id,
                "model": model_label,
                "rubric_type": job.rubric_type,
                "score": grade_result["score"],
                "earned_weight": grade_result["earned_weight"],
                "total_weight": grade_result["total_weight"],
                "criteria_results": grade_result["criteria_results"],
                "summary": grade_result["summary"],
                "time_seconds": round(elapsed, 3),
            }

            out.write_text(json.dumps(result, indent=2))
            log.info(f"[{desc}] score={result['score']:.4f} ({elapsed:.1f}s)")
            return {"task": task.task_id, "model": model_label,
                    "rubric_type": job.rubric_type, "status": "success"}

        except Exception as e:
            elapsed = time.time() - start
            error_result = {
                "task": task.task_id,
                "model": model_label,
                "rubric_type": job.rubric_type,
                "error": f"{type(e).__name__}: {e}",
                "time_seconds": round(elapsed, 3),
            }
            out.write_text(json.dumps(error_result, indent=2))
            log.error(f"[{desc}] FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
            return {"task": task.task_id, "model": model_label,
                    "rubric_type": job.rubric_type, "status": "error",
                    "error": str(e)}


# =============================================================================
# Verification + auto-retry pass
# =============================================================================

def find_failures(output_dir: Path, jobs: list[Job]) -> list[Job]:
    """Scan output dir and return jobs that failed or are missing."""
    failures = []
    for job in jobs:
        p = output_path(output_dir, job)
        if not p.exists():
            failures.append(job)
            continue
        try:
            data = json.loads(p.read_text())
            if "error" in data:
                error_msg = data["error"].lower()
                # Retry transient errors, skip permanent ones
                if any(t.lower() in error_msg for t in TRANSIENT_ERRORS):
                    p.unlink()  # Remove so it gets re-run
                    failures.append(job)
        except (json.JSONDecodeError, KeyError):
            failures.append(job)
    return failures


# =============================================================================
# Main orchestration
# =============================================================================

def get_task_list(
    tasks_dir: Path,
    task_id: str | None,
    tasks_file: Path | None,
) -> list[str]:
    """
    Resolve task list from (in priority order):
      1. --task-id  (single task)
      2. --tasks-file  (jsonl — same format as lighthouse: {"task_id": "..."} per line)
      3. Auto-discover from tasks_dir
    """
    if task_id:
        return [task_id]

    if tasks_file and tasks_file.exists():
        task_ids = []
        for line in tasks_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("task_id") or obj.get("id", "")
                if tid:
                    task_ids.append(tid)
            except json.JSONDecodeError:
                if line:
                    task_ids.append(line)
        return task_ids

    tasks = []
    for item in sorted(tasks_dir.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            if (item / "rubric" / "planning.json").exists():
                tasks.append(item.name)
    return tasks


async def run_all(args):
    if not args.tasks_dir.is_dir():
        log.error(f"Tasks directory not found or not a directory: {args.tasks_dir}")
        sys.exit(1)
    task_ids = get_task_list(args.tasks_dir, args.task_id, args.tasks_file)
    plan_models = [normalize_litellm_model(m) for m in args.plan_models.split(",") if m.strip()]
    run_execution = args.execution

    log.info(f"Tasks: {len(task_ids)} | Plan models: {plan_models} | "
             f"Execution grading: {run_execution} | Workers: {args.workers}")

    if not task_ids:
        log.error("No task IDs found. Check --tasks-dir and --tasks-file (or --task-id).")
        sys.exit(1)

    if args.dry_run:
        for t in task_ids:
            print(t)
        return

    # Load all tasks
    tasks: dict[str, TaskData] = {}
    for tid in task_ids:
        try:
            tasks[tid] = load_task(tid, args.tasks_dir)
        except Exception as e:
            log.error(f"Failed to load task {tid}: {e}")

    if not tasks:
        log.error("No tasks loaded (tasks-dir may be wrong or task dirs missing).")
        sys.exit(1)

    # Build job list
    all_jobs: list[Job] = []
    for tid in tasks:
        task = tasks[tid]

        # Write _metadata.json
        meta_dir = args.output_dir / tid
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "_metadata.json").write_text(json.dumps(task.metadata, indent=2))

        # Plan grading jobs
        if task.planning_rubric:
            if task.golden_plan:
                all_jobs.append(Job(tid, "golden", "planning", is_golden=True))
            for model in plan_models:
                all_jobs.append(Job(tid, model, "planning"))

        # Execution grading jobs
        if run_execution and task.execution_rubric:
            if task.golden_patch:
                all_jobs.append(Job(tid, "golden", "execution", is_golden=True))

    # Filter out already-complete jobs (resume support), unless overwrite
    if args.overwrite:
        pending = all_jobs
        skipped = 0
    else:
        pending = [j for j in all_jobs if not is_complete(args.output_dir, j)]
        skipped = len(all_jobs) - len(pending)
        if skipped:
            log.info(f"Resuming: {skipped} jobs already complete, {len(pending)} remaining")

    # Run with concurrency limit
    sem = asyncio.Semaphore(args.workers)
    start = time.time()

    async def _run_batch(jobs: list[Job]) -> list[dict]:
        coros = [
            run_job(j, tasks[j.task_id], args.output_dir, args.grade_model, sem)
            for j in jobs
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    results = await _run_batch(pending)

    # Count results
    succeeded = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failed = sum(1 for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") == "error"))

    log.info(f"Pass 1 done: {succeeded} succeeded, {failed} failed ({time.time() - start:.0f}s)")

    # ---- Verification + auto-retry loop ----
    max_retry_rounds = 3
    for retry_round in range(1, max_retry_rounds + 1):
        retry_jobs = find_failures(args.output_dir, all_jobs)
        if not retry_jobs:
            log.info("Verification passed — all jobs complete")
            break

        log.info(f"Retry round {retry_round}/{max_retry_rounds}: {len(retry_jobs)} jobs to retry")
        # Back off before retry round
        await asyncio.sleep(5 * retry_round)

        retry_results = await _run_batch(retry_jobs)
        retry_ok = sum(1 for r in retry_results if isinstance(r, dict) and r.get("status") == "success")
        retry_fail = sum(1 for r in retry_results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") == "error"))
        succeeded += retry_ok
        log.info(f"Retry round {retry_round}: {retry_ok} recovered, {retry_fail} still failing")
    else:
        remaining = find_failures(args.output_dir, all_jobs)
        if remaining:
            log.warning(f"{len(remaining)} jobs still failed after {max_retry_rounds} retry rounds")

    # ---- Final summary ----
    elapsed = time.time() - start

    final_ok = 0
    final_fail = 0
    final_skip = 0
    for j in all_jobs:
        p = output_path(args.output_dir, j)
        if not p.exists():
            final_fail += 1
        else:
            try:
                data = json.loads(p.read_text())
                if "error" in data:
                    final_fail += 1
                elif data.get("status") == "skipped":
                    final_skip += 1
                else:
                    final_ok += 1
            except json.JSONDecodeError:
                final_fail += 1

    log.info("=" * 60)
    log.info(f"COMPLETED in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log.info(f"  Succeeded: {final_ok}")
    log.info(f"  Skipped:   {final_skip + skipped}")
    log.info(f"  Failed:    {final_fail}")
    log.info(f"  Total:     {len(all_jobs)}")
    log.info("=" * 60)

    summary = {
        "plan_models": plan_models,
        "grade_model": args.grade_model,
        "execution_grading": run_execution,
        "total_tasks": len(task_ids),
        "total_jobs": len(all_jobs),
        "succeeded": final_ok,
        "skipped": final_skip + skipped,
        "failed": final_fail,
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(
        description="Multi-model P&C plan generation and rubric grading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("evals"))
    p.add_argument("--task-id", type=str, default=None, help="Single task")
    p.add_argument(
        "--tasks-file", type=Path, default=None,
        help='JSONL file with {"task_id": "..."} per line (same format as lighthouse)',
    )
    p.add_argument(
        "--plan-models",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="Comma-separated models for plan generation",
    )
    p.add_argument("--grade-model", type=str, default="openai/gpt-4o")
    p.add_argument(
        "--execution", action="store_true",
        help="Also grade golden patches against execution rubric",
    )
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification pass on existing results (no new jobs)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing result files instead of skipping complete jobs",
    )
    args = p.parse_args()

    if args.verify_only:
        task_ids = get_task_list(args.tasks_dir, args.task_id, args.tasks_file)
        plan_models = [normalize_litellm_model(m.strip()) for m in args.plan_models.split(",")]
        all_jobs = []
        for tid in task_ids:
            td = args.tasks_dir / tid
            if (td / "rubric" / "planning.json").exists():
                if (td / "golden_plan.md").exists():
                    all_jobs.append(Job(tid, "golden", "planning", is_golden=True))
                for m in plan_models:
                    all_jobs.append(Job(tid, m, "planning"))
            if args.execution and (td / "rubric" / "execution.json").exists():
                if (td / "golden.patch").exists():
                    all_jobs.append(Job(tid, "golden", "execution", is_golden=True))

        failures = find_failures(args.output_dir, all_jobs)
        complete = len(all_jobs) - len(failures)
        print(f"Complete: {complete}/{len(all_jobs)}")
        if failures:
            print(f"Failures ({len(failures)}):")
            for j in failures:
                model = "golden" if j.is_golden else j.model
                print(f"  {j.task_id}/{j.rubric_type}/{model}")
        sys.exit(1 if failures else 0)

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
