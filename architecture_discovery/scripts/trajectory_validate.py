"""Validate frozen trajectory inputs without producing analysis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_analysis.pipeline import load_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--allow-unannotated",
        action="store_true",
        help="engineering inspection only; scientific analysis requires double coding",
    )
    args = parser.parse_args()
    manifest, events, annotations, resolved, agreement, validation = load_study(
        args.manifest,
        args.data_root,
        require_annotations=not args.allow_unannotated,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "study_id": manifest.study_id,
                "runs": validation.run_count,
                "events": len(events),
                "raw_annotations": len(annotations),
                "resolved_annotations": len(resolved),
                "agreement": agreement,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
