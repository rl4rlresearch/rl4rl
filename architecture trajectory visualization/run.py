#!/usr/bin/env python3
"""Launch the architecture viewer or prepare/export its recorded trajectory data."""

from __future__ import annotations

import sys
from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FEATURE_ROOT.parent


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Keep the feature a normal Python package even though its containing
    # directory uses the human-readable name requested for the repository.
    for directory in (REPO_ROOT, FEATURE_ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    if args in (["--help"], ["-h"]):
        command = 'python "architecture trajectory visualization/run.py"'
        print(
            "Architecture trajectory visualization\n\n"
            f"  {command} serve [server options]\n"
            f"  {command} prepare [archive options]\n"
            f"  {command} [--repo PATH] catalog|export|prepare-archive [options]\n\n"
            "Use COMMAND --help for command-specific options."
        )
        return 0
    if not args or args[0] == "serve":
        from experiments.live_trajectory_dashboard import main as serve

        # Preserve the existing dashboard's direct-launch and hot-reload behavior.
        sys.argv = [
            str(REPO_ROOT / "experiments/live_trajectory_dashboard.py"), *args[1:]
        ]
        serve()
        return 0
    if args[0] == "prepare":
        from architecture_trajectory_visualization.prepare import main as prepare

        return prepare(args[1:])
    from architecture_trajectory_visualization.replay import main as replay

    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
