from __future__ import annotations

from experiments import live_trajectory_dashboard as dashboard


def test_stagnation_page_is_a_main_dashboard_tab() -> None:
    page = dashboard.STAGNATION_PAGE_PATH.read_text(encoding="utf-8")

    assert 'href="/">Trajectory explorer</a>' in page
    assert 'href="/ontology">Ontology explorer</a>' in page
    assert 'class="active" href="/stagnation">Stagnation &amp; breakthroughs</a>' in page
    assert "function summarizeRun" in page
    assert "raw_objective" in page
    assert "architecture or mechanism novelty" in page
    assert "/stagnation" in dashboard.read_dashboard_page()
    assert "/stagnation" in dashboard.read_science_page()
    assert "/stagnation" in dashboard.read_transcript_page()
    assert "/stagnation" in dashboard.read_controller_page()
