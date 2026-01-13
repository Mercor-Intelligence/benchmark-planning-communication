"""
Configuration for SWE-Bench Extended benchmark.

This module defines:
- SweBenchExtConfig: Benchmark-level configuration (extends BenchmarkConfig)
- SweBenchExtOptions: Runtime options for task execution

Author: Mercor Intelligence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field

from lighthouse.core.benchmark_tasks.benchmark_config import BenchmarkConfig, BenchmarkType


class SweBenchExtConfig(BenchmarkConfig):
    """
    Configuration for SWE-Bench-Ext benchmark.
    
    Extends BenchmarkConfig from eval-framework for compatibility.
    Defines where tasks are stored and how to access them.
    """
    
    # Benchmark identity
    name: str = Field(default="swe-bench-ext", description="Benchmark name")
    type: str = Field(default="code", description="Benchmark type (code/terminal)")
    workdir: str = Field(default="/workspace/repo", description="Working directory in sandbox")
    
    excluded_context: List[str] = Field(default_factory=list, description="Context to exclude from prompts")