"""Controlled subprocess environment shared by agents and evaluators."""

from __future__ import annotations

import os


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
