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

Set API keys in `.env` for the providers you use (OpenAI, Anthropic, Google, Fireworks, ByteDance); see `.env.example`. For multi-model plan grading with custom endpoints, use `config/plan_grading_models.yaml` (see Plan Generation & Grading below).

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

### Execution grading

- **From the harness:** When you run `lighthouse execute-single` or `execute-batch`, the harness runs the execution rubric grader after each run and stores the result (e.g. `rubric_grade_summary`) in the run output. Execution results live in lighthouse’s timestamped output dir (e.g. `test-evals/2026-02-25_18-30-00/`). Use that output as the source of truth for model execution grades.

- **Grade golden execution only (no harness):** To grade only the reference solution (`golden.patch`) against the execution rubric, use plan grading with `--execution`. That writes `output-dir/<task_id>/execution/golden.json` in the same format as plan grading (score, criteria_results, summary):
  ```bash
  uv run python scripts/run_plan_grading.py --tasks-dir /path/to/tasks --output-dir evals --execution --grade-model openai/gpt-4o
  ```
  This does not grade model-generated solutions; for those, run the execution harness and use the grades produced there.

### Grading Solutions

```bash
lighthouse grade-single \
    --benchmark planning_communication \
    --task-id my-task-123 \
    --solution-file solution.patch \
    --task-source-file config/task_source_config.yaml
```

## Plan Generation & Grading (Separate)

Plan grading runs independently of execution — outside the lighthouse harness. For each task, the script generates plans with the requested models, grades them (and the golden plan when present) against the planning rubric, and writes one JSON per task/model containing `score`, `criteria_results`, `summary`, and `generated_plan`.

**Run with the project environment** so `planning_communication` and `litellm` are available:

```bash
# Option A: use uv (recommended)
uv run python scripts/run_plan_grading.py ...

# Option B: activate venv then run
source .venv/bin/activate
python scripts/run_plan_grading.py ...
```

**Using a models config (recommended):** Use `--models-config config/plan_grading_models.yaml` to run all models in the YAML with their endpoints and API keys. Omit `--plan-models` to use every key in the config, or pass `--plan-models claude-sonnet-4.5,kimi-k2-thinking` to run a subset.

Examples:

```bash
# Single task, one plan model
uv run python scripts/run_plan_grading.py \
    --tasks-dir /path/to/tasks \
    --task-id my-task-123 \
    --plan-models anthropic/claude-sonnet-4-5-20250929 \
    --grade-model openai/gpt-4o

# All tasks, all models from YAML (custom endpoints for Kimi, ByteDance, etc.)
uv run python scripts/run_plan_grading.py \
    --tasks-dir /path/to/tasks \
    --output-dir plan-grading-evals \
    --models-config config/plan_grading_models.yaml \
    --grade-model openai/gpt-4o \
    --workers 50

```

Output layout: `output-dir/<task_id>/planning/<model>.json` (and `output-dir/<task_id>/execution/golden.json` when using `--execution`). Each planning JSON includes `generated_plan` (the plan that was graded). Use `--overwrite` to regenerate and overwrite existing result files.

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
├── config/                          # Config examples + plan_grading_models.yaml
├── module_config.yaml               # Registers with lighthouse plugin system
├── pyproject.toml
└── README.md
```
