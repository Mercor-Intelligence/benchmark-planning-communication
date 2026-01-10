# benchmark-swe-bench-ext

SWE-Bench Extended benchmark task implementation for [eval-framework](https://github.com/Mercor-Intelligence/eval-framework).

## Setup

```bash
git clone --recursive https://github.com/Mercor-Intelligence/benchmark-swe-bench-ext.git
cd benchmark-swe-bench-ext
pip install -e .
```

## Usage

```python
from swe_bench_ext import SweBenchExtTask, SweBenchExtConfig

# Load task from local path
task = SweBenchExtTask.from_local_path(Path("/path/to/task"))

# Get prompts
system_prompt = task.get_system_prompt()
user_prompt = task.get_initial_user_prompt()

# Get scripts
grading_scripts = task.generate_grading_setup_script()
test_script = task.generate_test_run_script()

# Parse results
summary = task.parse_test_results(test_output)
```

## Structure

```
benchmark-swe-bench-ext/
├── swe_bench_ext/
│   ├── __init__.py    # Package exports
│   ├── task.py        # SweBenchExtTask
│   └── config.py      # Config & Options
├── eval-framework/    # Git submodule
├── pyproject.toml
└── README.md
```
