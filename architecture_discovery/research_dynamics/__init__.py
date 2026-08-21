"""Experimental instrumentation for research-process intervention studies.

This package treats existing discovery controllers as experimental subjects.  It
changes only controller-visible memory and deliberation instructions.  Search
selection, evaluation, retention, budgets, and stopping remain owned by the
existing controllers.
"""

from research_dynamics.contracts import (
    ConditionId,
    DeliberationPolicy,
    FrameworkKind,
    ProcessStudyConfig,
    VisibleMemoryPolicy,
)
from research_dynamics.protocol import ProcessProtocol

__all__ = [
    "ConditionId",
    "DeliberationPolicy",
    "FrameworkKind",
    "ProcessProtocol",
    "ProcessStudyConfig",
    "VisibleMemoryPolicy",
]
