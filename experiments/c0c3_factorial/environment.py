"""Controlled subprocess environment shared by agents and evaluators."""

from __future__ import annotations

import os
from pathlib import Path


def controlled_subprocess_environment(run_seed: int | None) -> dict[str, str]:
    """Return an inherited environment with the frozen block seed exposed.

    ``C0C3_RUN_SEED`` is the framework-neutral task seed. ``PYTHONHASHSEED``
    removes one common source of Python process nondeterminism. These controls
    cannot seed provider-side Codex model sampling; that limitation is recorded
    in the protocol documentation rather than hidden behind a false claim of
    deterministic model generations.
    """

    environment = os.environ.copy()
    if run_seed is None:
        return environment
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise ValueError("run_seed must be an integer")
    environment["C0C3_RUN_SEED"] = str(run_seed)
    environment["PYTHONHASHSEED"] = str(run_seed % (2**32))
    return environment


def subject_subprocess_environment(
    run_seed: int | None,
    *,
    workspace: str | Path | None = None,
    expose_run_seed: bool = True,
) -> dict[str, str]:
    """Build the subject environment without exposing internal seed labels."""

    environment = controlled_subprocess_environment(run_seed)
    environment.pop("C0C3_RUN_SEED", None)
    environment.pop("OLDPWD", None)
    if workspace is not None:
        subject_workspace = Path(workspace)
        environment["PWD"] = str(subject_workspace)
        if not expose_run_seed:
            cache = subject_workspace / ".subject-cache"
            cache.mkdir(parents=True, exist_ok=True)
            environment["TMPDIR"] = str(cache)
            environment["XDG_CACHE_HOME"] = str(cache)
    if run_seed is not None and expose_run_seed:
        environment["OPTIMIZATION_RUN_SEED"] = str(run_seed)
    else:
        environment.pop("OPTIMIZATION_RUN_SEED", None)
        environment.pop("PYTHONHASHSEED", None)
    return environment
