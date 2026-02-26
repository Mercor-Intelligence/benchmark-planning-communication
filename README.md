# Planning & Communication Harness

Planning & Communication benchmark task for [lighthouse](https://github.com/Mercor-Intelligence/lighthouse), extending [benchmark-swe-bench-ext](https://github.com/Mercor-Intelligence/benchmark-swe-bench-ext).

**Two independent evaluation flows:**

| Flow | What it evaluates | How it runs |
|------|-------------------|-------------|
| **Execution** | Agent code solutions | `lighthouse execute-single/batch` → sandbox → tests → execution rubric |
| **Plan grading** | Generated plans | `scripts/run_plan_grading.py` → LLM generates plan → planning rubric |

These are decoupled. Plan generation does not feed into execution context.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Mercor-Intelligence/benchmark-planning-communication.git
cd benchmark-planning-communication
```

Install with the `benchmark-swe-bench-ext` dependency (which brings `lighthouse` transitively):

```bash
uv pip install -e .

# SSH or local clone: install benchmark-swe-bench-ext first, then uv pip install -e .
```

Set up environment:

```bash
cp .env.example .env
cp config/task_source_config.yaml.example config/task_source_config.yaml
cp config/benchmark_task_config.yaml.example config/benchmark_task_config.yaml
```

## Task Structure

P&C tasks extend the standard SWE-Bench-Ext layout with planning-specific files:

```
tasks/
└── <task-name>/
    ├── Dockerfile
    ├── test_metadata.json
    ├── problem_statement.md
    ├── planning_statement.md      # P&C-specific
    ├── golden_plan.md             # P&C-specific
    ├── golden.patch
    ├── test.patch
    ├── requirements.json
    ├── interface.md
    └── rubric/
        ├── planning.json          # P&C-specific
        └── execution.json         # P&C-specific
```

## Execution Harness Evals

Identical to SWE-Bench-Ext. The agent gets a problem, solves it in a sandbox, code is tested, and optionally graded against the execution rubric.

```bash
# Single task
lighthouse execute-single \
    --benchmark planning_communication \
    --task-id my-task-123 \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml

# With thinking mode
lighthouse execute-single \
    --benchmark planning_communication \
    --task-id my-task-123 \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --thinking-mode \
    --task-source-file config/task_source_config.yaml

# Batch
lighthouse execute-batch \
    --benchmark planning_communication \
    --tasks-file tasks.jsonl \
    --model openai/o3-mini \
    --task-source-file config/task_source_config.yaml
```

Execution runs use the Modal sandbox by default (`--sandbox-type modal`). Lighthouse **automatically runs execution rubric grading** after each run and includes rubric scores in the result. Use `--output-dir` (base dir you use for plan grading) so all evals live under one tree; lighthouse creates a timestamped subdir (e.g. `test-evals/2026-02-25_18-30-00/`) so runs don’t overwrite each other.

### Execution grading on trajectories

- **Merge run into evals tree:** Copy lighthouse's timestamped run into your evals tree so each task has `target/<task_id>/execution/<model>.json`:
  ```bash
  uv run python scripts/merge_execution_results.py --source test-evals/2026-02-26_02-58-14 --target test-evals
  ```
  With `--watch` you can point `--source` at the parent dir and merge results as they appear.

- **Grade golden execution only:** Use plan grading with execution mode (reads `golden.patch` and writes `execution/golden.json`):
  ```bash
  uv run python scripts/run_plan_grading.py --tasks-dir tasks/ --output-dir test-evals --execution --mode golden-only --grade-model openai/gpt-4o
  ```

- **Grade model runs when lighthouse didn't:** If the streamed result has no `rubric_grade_summary` (e.g. Modal didn't run the grader), run the merge script with `--grade-missing` so it grades locally when merging and writes the same summary format as `golden.json`:
  ```bash
  uv run python scripts/merge_execution_results.py --source test-evals/2026-02-26_02-58-14 --target test-evals \
    --grade-missing --tasks-dir /path/to/tasks --grade-model openai/gpt-4o
  ```

### Grading Solutions

```bash
lighthouse grade-single \
    --benchmark planning_communication \
    --task-id my-task-123 \
    --solution-file solution.patch \
    --task-source-file config/task_source_config.yaml
```

## Plan Generation & Grading (Separate)

Plan grading runs independently of execution — outside the lighthouse harness.

**Run with the project environment** so `planning_communication` and `litellm` are available:

```bash
# Option A: use uv (recommended)
uv run python scripts/run_plan_grading.py ...

# Option B: activate venv then run
source .venv/bin/activate
python scripts/run_plan_grading.py ...
```

Examples:

```bash
# Grade the golden plan for a single task
uv run python scripts/run_plan_grading.py \
    --tasks-dir tasks/ \
    --task-id my-task-123 \
    --mode golden-only \
    --grade-model openai/gpt-4o

# Generate plans with a model and grade them
uv run python scripts/run_plan_grading.py \
    --tasks-dir tasks/ \
    --mode both \
    --plan-model google/gemini-2.5-pro \
    --grade-model openai/gpt-4o

# Batch all tasks
uv run python scripts/run_plan_grading.py \
    --tasks-dir tasks/ \
    --output-dir grading-results/ \
    --workers 20
```

## Rubric Format

Both `rubric/planning.json` and `rubric/execution.json` use the same category-keyed format:

```json
{
  "functional": [
    {
      "id": "functional-1",
      "description": "Plan addresses core requirement X",
      "weight": "major"
    }
  ],
  "robustness": [...],
  "style": [...]
}
```

Execution rubrics may also include a `"correctness"` category.

Scoring: major criteria = 2x weight, minor = 1x weight. Score = earned / total.

## Project Structure

```
benchmark-planning-communication/
├── planning_communication/          # Python package
│   ├── __init__.py
│   ├── task.py                      # PlanningCommTask (extends SweBenchExtTask)
│   ├── config.py                    # PlanningCommConfig (extends SweBenchExtConfig)
│   ├── planning_rubric_grader.py    # Evaluates plans against planning rubric
│   └── execution_rubric_grader.py   # Evaluates code against execution rubric
├── scripts/
│   └── run_plan_grading.py          # Standalone plan generation + grading
├── config/                          # Config examples
├── module_config.yaml               # Registers with lighthouse plugin system
├── pyproject.toml
└── README.md
```
