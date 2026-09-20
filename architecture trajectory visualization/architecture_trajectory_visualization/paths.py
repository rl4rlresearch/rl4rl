"""Locations are anchored to the checkout, independent of the working directory."""

from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FEATURE_ROOT.parent
VIEWER_ROOT = FEATURE_ROOT / "frontend"
