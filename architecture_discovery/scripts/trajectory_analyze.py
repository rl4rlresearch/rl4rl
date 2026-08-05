"""Run the frozen autonomous-research trajectory analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_analysis.pipeline import analyze_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-unannotated",
        action="store_true",
        help="engineering inspection only; marks missing edit labels unclassified",
    )
    args = parser.parse_args()
    provenance = analyze_study(
        args.manifest,
        args.data_root,
        args.output_dir,
        require_annotations=not args.allow_unannotated,
    )
    print(json.dumps(provenance, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
