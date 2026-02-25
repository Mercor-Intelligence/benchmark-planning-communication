# benchmark-swe-bench-ext

SWE-Bench Extended benchmark task implementation for [lighthouse](https://github.com/Mercor-Intelligence/lighthouse).

## Quick Start

### 1. Installation

We recommend [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone --recursive https://github.com/Mercor-Intelligence/benchmark-swe-bench-ext.git
cd benchmark-swe-bench-ext
```

Install the package along with the `lighthouse` dependency using one of the methods below:

```bash
# Via HTTPS (recommended for most users)
uv pip install -e ".[lighthouse_https]"

# Via SSH (if you have SSH keys configured for GitHub)
uv pip install -e ".[lighthouse_ssh]"
```

**If you don't have access to the private GitHub repo**, place a copy of the `lighthouse` repository in a `lighthouse/` subdirectory at the project root, then install from the local path:

```bash
uv pip install -e ".[lighthouse_local]"
```

> **Note:** Because `uv run` attempts to resolve all optional dependencies (including the remote ones) when generating a lockfile, you should use the `--no-project` flag to skip project resolution:
> ```bash
> uv run --no-project lighthouse execute-single ...
> ```
> Alternatively, activate the virtual environment directly and run `lighthouse` without `uv run`.

Then set up your environment variables:

```bash
cp .env.example .env
# Edit .env and fill in values needed for local execution
```

### 2. Set Up a Local Task Source

Copy the example configuration files:

```bash
cp config/task_source_config.yaml.example config/task_source_config.yaml
cp config/benchmark_task_config.yaml.example config/benchmark_task_config.yaml
```

The default `config/task_source_config.yaml` points to a local `tasks/` directory:

```yaml
source_type: local_folder
config:
  path: tasks/
  build_image_if_not_exists: true
```

Place your task folders under `tasks/`. Each task lives in its own subdirectory:

```
tasks/
├── my-org-my-repo-123/
│   ├── Dockerfile                 # Environment setup (required)
│   ├── test_metadata.json         # Test configuration (required)
│   ├── problem_statement.md       # Problem description
│   ├── prompt_statement.md        # Agent prompt (optional, falls back to problem_statement)
│   ├── golden.patch               # Reference solution
│   ├── test.patch                 # Test patch applied before grading
│   ├── requirements.json          # List of requirements
│   ├── interface.md               # Interface specifications (optional)
│   └── knowledge_base.md          # Additional context (optional)
```

With `build_image_if_not_exists: true`, lighthouse will automatically build Docker images from each task's `Dockerfile` when needed.

### 3. Validate Your Tasks

Before running full evaluations, use the validation commands to verify that your tasks are set up correctly. Validation grades two known solutions for each task and checks that:

- An **empty solution** (no changes) scores **0.0**
- The **golden solution** (`golden.patch`) scores **1.0**

Validate a single task:

```bash
uv run lighthouse validate-single \
    --benchmark swe_bench_ext \
    --task-id my-org-my-repo-123 \
    --task-source-file config/task_source_config.yaml
```

Or validate all tasks in a batch using a `tasks.jsonl` file:

```jsonl
{"task_id": "my-org-my-repo-123"}
{"task_id": "my-org-my-repo-456"}
```

```bash
uv run lighthouse validate-batch \
    --benchmark swe_bench_ext \
    --tasks-file tasks.jsonl \
    --task-source-file config/task_source_config.yaml
```

Validation supports additional options:

| Option | Default | Description |
|--------|---------|-------------|
| `--test-timeout` | 600 | Timeout for test execution in seconds |
| `--num-epochs` | 1 | Number of validation runs per task |
| `--sandbox-type` | modal | Sandbox type (`modal` or `docker`) |
| `--output-dir` | logs | Directory to write validation results |

### 4. Run Your First Evaluation

Once validation passes, run an agent evaluation:

```bash
uv run lighthouse execute-single \
    --benchmark swe_bench_ext \
    --task-id my-org-my-repo-123 \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --task-source-file config/task_source_config.yaml \
    --benchmark-overrides-file config/benchmark_task_config.yaml
```

> **Note:** If using uv, you can run commands with `uv run lighthouse ...`. Alternatively, activate your virtual environment first (`source .venv/bin/activate`) and run `lighthouse ...` directly.

### 5. Configure Benchmark (Optional)

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

### 6. Configure Tools (Optional)

The agent has access to a default set of tools. You can customize which tools are available:

**Default tools:** `bash`, `view_file`, `str_replace`, `create_file`, `insert_str`

```bash
# Add tools to the defaults (e.g., add ask_question for clarifying questions)
uv run lighthouse execute-single ... --add-tools ask_question

# Or replace the default tools entirely
uv run lighthouse execute-single ... --tools bash view_file str_replace
```

You can also configure tool-specific options via a YAML file:

```yaml
# config/tool_options.yaml
bash:
  timeout: 120
  max_output_lines: 1000
view_file:
  max_lines: 500
```

Then pass it to the CLI:

```bash
uv run lighthouse execute-single ... --tool-options-file config/tool_options.yaml
```

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

## Validating Tasks

Validation checks that a task's test harness is working correctly by grading two known solutions: an empty patch (expected score: 0.0) and the golden patch (expected score: 1.0). This is the recommended first step before running agent evaluations.

### Validate a Single Task

```bash
uv run lighthouse validate-single \
    --benchmark swe_bench_ext \
    --task-id my-org-my-repo-123 \
    --task-source-file config/task_source_config.yaml
```

### Validate Multiple Tasks (Batch)

```bash
uv run lighthouse validate-batch \
    --benchmark swe_bench_ext \
    --tasks-file tasks.jsonl \
    --task-source-file config/task_source_config.yaml
```

Results are written to the `--output-dir` (default: `logs/`) with per-task validation details and a `summary.json`.

### Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-timeout` | 600 | Test execution timeout in seconds |
| `--num-epochs` | 1 | Number of validation runs per task (for flakiness detection) |
| `--sandbox-type` | modal | Sandbox type (`modal` or `docker`) |
| `--output-dir` | logs | Directory to write validation results |
| `--concurrency-limit` | (no limit) | Maximum concurrent validations |

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

# Validate a single task using Docker
uv run lighthouse validate-single \
    --benchmark swe_bench_ext \
    --task-id my-org-my-repo-123 \
    --task-source-file config/task_source_config.yaml \
    --sandbox-type docker

# Validate all tasks in batch with concurrency limit
uv run lighthouse validate-batch \
    --benchmark swe_bench_ext \
    --tasks-file tasks.jsonl \
    --task-source-file config/task_source_config.yaml \
    --concurrency-limit 5
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
