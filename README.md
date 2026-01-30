# benchmark-swe-bench-ext

SWE-Bench Extended benchmark task implementation for [lighthouse](https://github.com/Mercor-Intelligence/lighthouse).

## Quick Start

### 1. Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone --recursive https://github.com/Mercor-Intelligence/benchmark-swe-bench-ext.git
cd benchmark-swe-bench-ext
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and fill in values needed for local execution
```

### 2. Configure Task Source

Copy the example configuration files:

```bash
cp config/task_source_config.yaml.example config/task_source_config.yaml
cp config/benchmark_task_config.yaml.example config/benchmark_task_config.yaml
```

Edit `config/task_source_config.yaml` to point to your tasks:

```yaml
source_type: local_folder
config:
  path: tasks/
  build_image_if_not_exists: true
```

### 3. Configure Benchmark (Optional)

Edit `config/benchmark_task_config.yaml` to customize benchmark behavior. The defaults work out of the box, but you can adjust these options:

```yaml
# Exclude context sections from prompts (options: problem_statement, requirements, interface, knowledge_base)
excluded_context: []

# Use pre-built Docker images from a registry (optional)
# image_uri_template: "your-registry.com/repo:{task_id}"

# Enable PR artifacts generation stage
enable_pr_artifacts: false

# Configure reminder behavior (modes: "off", "constant")
reminder_policy:
  readme_mode: "off"
  ask_question_mode: "off"
  artifact_mode: "off"
```

### 4. Run Your First Evaluation

```bash
uv run lighthouse execute-single \
    --benchmark swe_bench_ext \
    --task-id django__django-12345 \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml \
    --benchmark-overrides-file config/benchmark_task_config.yaml
```

> **Note:** All `lighthouse` commands should be run with `uv run lighthouse` to use the uv-managed environment, or activate the virtual environment first with `source .venv/bin/activate`.

## Running Evaluations

### Execute a Single Task

```bash
uv run lighthouse execute-single \
    --benchmark swe_bench_ext \
    --task-id django__django-12345 \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml
```

### Execute Multiple Tasks (Batch)

Create a `tasks.jsonl` file:

```jsonl
{"task_id": "django__django-12345"}
{"task_id": "astropy__astropy-67890", "image_override": "custom-image:latest"}
```

Run batch execution:

```bash
uv run lighthouse execute-batch \
    --benchmark swe_bench_ext \
    --tasks-file tasks.jsonl \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml
```

### Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Model identifier (e.g., `anthropic/claude-sonnet-4-5-20250929`) |
| `--max-steps` | 100 | Maximum agent steps |
| `--max-tokens` | 128000 | Maximum total tokens |
| `--temperature` | 0.0 | Sampling temperature |
| `--tools` | bash, read_file, write_file | Tools available to the agent |
| `--num-epochs` | 1 | Number of trials per task (for pass@k) |
| `--sandbox-type` | modal | Sandbox type (`modal` or `docker`) |
| `--sandbox-timeout` | 3600 | Maximum sandbox lifetime in seconds |

## Grading Solutions

### Grade a Single Solution

```bash
uv run lighthouse grade-single \
    --benchmark swe_bench_ext \
    --task-id django__django-12345 \
    --solution-file solution.patch \
    --task-source-file config/task_source_config.yaml
```

Optionally include the agent trajectory for analysis:

```bash
uv run lighthouse grade-single \
    --benchmark swe_bench_ext \
    --task-id django__django-12345 \
    --solution-file solution.patch \
    --trajectory-file trajectory.json \
    --task-source-file config/task_source_config.yaml
```

### Grade Multiple Solutions (Batch)

Create a `predictions.jsonl` file:

```jsonl
{"task_id": "django__django-12345", "solution": "diff --git a/..."}
{"task_id": "astropy__astropy-67890", "solution": "diff --git b/...", "trajectory": "..."}
```

Run batch grading:

```bash
uv run lighthouse grade-batch \
    --benchmark swe_bench_ext \
    --predictions-file predictions.jsonl \
    --task-source-file config/task_source_config.yaml
```

### Grading Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-timeout` | 600 | Test execution timeout in seconds |
| `--run-rubric` / `--no-run-rubric` | true | Enable/disable LLM-based rubric grading |
| `--rubric-model` | (eval model) | Model for rubric grading |
| `--analyze-trajectory` / `--no-analyze-trajectory` | true | Enable/disable trajectory analysis |
| `--derive-failure-mode` / `--no-derive-failure-mode` | true | Enable/disable failure mode derivation |

## Common Commands

```bash
# View help
uv run lighthouse --help
uv run lighthouse execute-single --help

# Execute with verbose logging
uv run lighthouse execute-single \
    --benchmark swe_bench_ext \
    --task-id my-task \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml \
    -v

# Execute with concurrency limit
uv run lighthouse execute-batch \
    --benchmark swe_bench_ext \
    --tasks-file tasks.jsonl \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml \
    --concurrency-limit 5

# Grade with custom test timeout
uv run lighthouse grade-batch \
    --benchmark swe_bench_ext \
    --predictions-file predictions.jsonl \
    --task-source-file config/task_source_config.yaml \
    --test-timeout 1200
```

---

## Reference

### Configuration Files

#### config/task_source_config.yaml

Configures where tasks are loaded from:

```yaml
source_type: local_folder
config:
  path: tasks/
  build_image_if_not_exists: true
```

For S3-based tasks:

```yaml
source_type: s3_zip
config:
  path: s3://my-bucket/tasks
  local_cache_dir: /tmp/task_cache
```

#### config/benchmark_task_config.yaml

Benchmark-specific configuration:

```yaml
excluded_context: []  # List of context sections to exclude from prompts
                      # Options: problem_statement, requirements, interface, knowledge_base
```

### Task Directory Structure

By default, tasks are loaded from the `tasks/` directory (configured in `config/task_source_config.yaml`). Each task should be in its own subdirectory:

```
tasks/
├── django__django-12345/
│   ├── test_metadata.json      # Test configuration (required)
│   ├── problem_statement.md    # Problem description
│   ├── prompt_statement.md     # Customized prompt (optional, falls back to problem_statement)
│   ├── golden.patch            # Reference solution
│   ├── test.patch              # Test patch to apply before grading
│   ├── requirements.json       # List of requirements
│   ├── interface.md            # Interface specifications (optional)
│   └── knowledge_base.md       # Additional context (optional)
├── astropy__astropy-67890/
│   └── ...
```

#### test_metadata.json Format

```json
{
  "language": "python",
  "test_framework": "pytest",
  "test_command": "pytest tests/ -xvs",
  "test_files": ["tests/test_example.py"],
  "FAIL_TO_PASS": ["tests/test_example.py::test_bug_fix"],
  "PASS_TO_PASS": ["tests/test_example.py::test_existing"],
  "base_commit": "abc123"
}
```

### Python API Usage

```python
from swe_bench_ext import SweBenchExtTask, SweBenchExtConfig
from lighthouse.task_source.local_folder import LocalFolderTaskSource

# Load task from task source
task_source = LocalFolderTaskSource(path="tasks/")
task = SweBenchExtTask.from_id("django__django-12345", task_source)

# Get prompts
system_prompt = task.get_system_prompt()
user_prompt = task.get_initial_user_prompt()

# Get scripts
setup_script = task.generate_setup_script()
grading_scripts = task.generate_grading_setup_script()
test_script = task.generate_test_run_script()

# Parse results
summary = task.parse_test_results(test_output)
print(f"Score: {summary.score}")
print(f"Passed: {summary.metadata['f2p_passed']}/{summary.metadata['f2p_total']}")
```

#### Rubric-Based Grading

```python
from swe_bench_ext import SweBenchExtTask, SweBenchExtRubricGrader
from task_source.local_folder import LocalFolderTaskSource

# Load task and rubric
task_source = LocalFolderTaskSource(path="/path/to/tasks")
task = SweBenchExtTask.from_id("libgeos-geos-1182-1239", task_source)
rubric = task.load_rubric(task_source)

# Create grader with OpenAI
grader = SweBenchExtRubricGrader(
    rubric=rubric,
    model_name="openai/gpt-4o-mini",
    api_key="your-api-key",
)

# Or with Anthropic
grader = SweBenchExtRubricGrader(
    rubric=rubric,
    model_name="anthropic/claude-3-5-sonnet-20241022",
    api_key="your-api-key",
)

# Grade solution
result = await grader.grade(
    solution="Fixed the bug by...",
    git_diff=git_diff,
    problem_statement=task.problem_statement,
    trajectory={"transcript": agent_conversation},
)

print(f"Total Score: {result.total_score}")
print(f"Explanation:\n{result.explanation}")

# Access individual criteria scores
for score in result.criteria_scores:
    print(f"{score.criteria.criteria_id}: {score.score} - {score.explanation}")
```

### Rubric Format

Rubrics are organized by category (functional, robustness, style, etc.):

```python
# Access rubric categories
categories = rubric.get_all_categories()  # ['functional', 'robustness', 'style']

# Get criteria by category
functional_criteria = rubric.get_criteria_by_category('functional')

# Supports both numeric weights and "major"/"minor" labels
# major = 1.0, minor = 0.5
```

### Features

- **Task Loading** - Load tasks from local directories or S3  
- **Prompt Generation** - System and user prompts for agents  
- **Test Execution** - Setup and run test scripts  
- **Result Parsing** - Parse test output into structured summaries  
- **Rubric Grading** - LLM-based evaluation against structured rubrics

### Project Structure

```
benchmark-swe-bench-ext/
├── swe_bench_ext/
│   ├── __init__.py           # Package exports
│   ├── task.py               # SweBenchExtTask
│   ├── config.py             # Config & Options
│   ├── rubric_grader.py      # LLM-based rubric grader
│   └── rubric_utils.py       # Rubric format conversion
├── eval-framework/           # Git submodule
│   ├── __init__.py         # Package exports
│   ├── task.py             # SweBenchExtTask implementation
│   └── config.py           # SweBenchExtConfig
├── config/
│   ├── benchmark_task_config.yaml   # Benchmark configuration
│   └── task_source_config.yaml      # Task source configuration
├── tasks/                  # Default task directory
│   └── <task-id>/          # Individual task folders
├── lighthouse/             # Git submodule (eval framework)
├── pyproject.toml
└── README.md
```

### Dependencies

- **eval-framework** (submodule) - Core abstractions
- **openai>=1.0.0** (optional) - For OpenAI grading
- **anthropic>=0.18.0** (optional) - For Anthropic grading

## License

See [LICENSE](LICENSE) file for details.
