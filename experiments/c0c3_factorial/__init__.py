"""Controlled C0-C3 autonomous-research experiments.

The package is intentionally isolated from the older exploratory runners.  Its
public modules use only the Python standard library so campaign creation,
validation, and analysis do not require a GPU environment.
"""

from .spec import Condition, FactorialSpec, FrameworkSpec, TaskSpec

__all__ = ["Condition", "FactorialSpec", "FrameworkSpec", "TaskSpec"]
