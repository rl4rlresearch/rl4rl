#!/usr/bin/env python3
"""Repository entry point for research-process experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_dynamics.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
