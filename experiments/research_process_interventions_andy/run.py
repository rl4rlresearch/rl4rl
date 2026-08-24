#!/usr/bin/env python3
"""Repository entry point for research-process experiments."""

import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "architecture_discovery"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


def main() -> int:
    """Delegate to the shared research-dynamics CLI."""

    from research_dynamics.cli import main as research_dynamics_main

    return research_dynamics_main()


if __name__ == "__main__":
    raise SystemExit(main())
