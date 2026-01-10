"""
SWE-Bench Extended Benchmark Task Implementation.

This package provides the benchmark-specific implementation for SWE-Bench-Ext tasks,
extending the eval-framework's BaseBenchmarkTask.

Usage:
    from swe_bench_ext import SweBenchExtTask, SweBenchExtConfig
    
    # Load a task
    task = SweBenchExtTask.from_task_source(task_source, "0xpolygon-bor-1710")
    
    # Get prompts for agent
    system_prompt = task.get_system_prompt()
    user_prompt = task.get_initial_user_prompt()
    
    # Get scripts for execution
    setup_scripts = task.generate_setup_script()
    grading_scripts = task.generate_grading_setup_script()
    test_script = task.generate_test_run_script()
    
    # Parse results
    summary = task.parse_test_results(test_output)
"""

from .task import SweBenchExtTask
from .config import SweBenchExtConfig, SweBenchExtOptions

__all__ = [
    "SweBenchExtTask",
    "SweBenchExtConfig",
    "SweBenchExtOptions",
]

__version__ = "0.1.0"
