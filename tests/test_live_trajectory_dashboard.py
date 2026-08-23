from __future__ import annotations

from pathlib import Path

from experiments.live_trajectory_dashboard import dashboard_data


def test_dashboard_data_keeps_legacy_refresh_keys(tmp_path: Path) -> None:
    autoresearch = tmp_path / "autoresearch"
    openevolve = tmp_path / "openevolve"
    autoresearch.mkdir()
    openevolve.mkdir()
    prices = {"input": 1.75, "cached_input": 0.175, "output": 14.0}

    data = dashboard_data(
        {
            "autoresearch_v16": autoresearch,
            "openevolve_v2": openevolve,
            "autoresearch_v17": tmp_path / "not-started",
        },
        prices,
    )

    assert data["schema_version"] == "2.0"
    assert data["autoresearch"] == data["campaigns"]["autoresearch_v16"]
    assert data["openevolve_v2"] == data["campaigns"]["openevolve_v2"]
    assert data["campaigns"]["autoresearch_v17"]["available"] is False
