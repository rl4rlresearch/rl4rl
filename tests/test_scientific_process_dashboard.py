from __future__ import annotations

import json
from pathlib import Path

from experiments.live_trajectory_dashboard import (
    SCIENCE_PAGE_PATH,
    manipulation_review_index,
    substantive_claim,
)


def test_science_page_is_separate_and_process_focused() -> None:
    page = SCIENCE_PAGE_PATH.read_text(encoding="utf-8")
    assert 'href="/">Trajectory explorer</a>' in page
    assert 'class="active" href="/science">Scientific process</a>' in page
    assert "Scientific process observatory" in page
    assert "Outcome quality is deliberately secondary" in page
    assert "Cumulative distinct source deltas" in page
    assert "Post-intervention phase response" in page
    assert "Intervention evidence dossier" in page
    assert "function renderAudit(payload)" in page
    assert "function renderPhaseTable(payload)" in page


def test_substantive_claim_rejects_missing_markers() -> None:
    assert substantive_claim("A falsifiable architectural hypothesis")
    assert not substantive_claim("[not recorded]")
    assert not substantive_claim(" [missing mechanism] ")
    assert not substantive_claim(None)


def test_manipulation_review_index_preserves_pending_annotations(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    review = campaign / "review/v3-manipulation"
    packets = review / "packets"
    packets.mkdir(parents=True)
    mapping = {
        "packet_id": "packet-a",
        "run_id": "run-a",
        "opportunity": 6,
        "proposal_type": "assumption_changing",
    }
    (review / "private-mapping.jsonl").write_text(
        json.dumps(mapping) + "\n", encoding="utf-8"
    )
    (packets / "packet-a.json").write_text(
        json.dumps(
            {
                "annotation": {
                    "old_assumption_identifiable": True,
                    "new_mechanism_implemented": None,
                    "distinct_from_recent_lineage": None,
                    "primarily_tuning_pruning_or_deletion": None,
                    "cleanly_attributable": None,
                    "feasible_under_task_contract": None,
                    "novelty_score": None,
                }
            }
        ),
        encoding="utf-8",
    )

    index, summary = manipulation_review_index(campaign)
    assert index[("run-a", 6)]["reviewed"] is True
    assert index[("run-a", 6)]["fully_reviewed"] is False
    assert summary["packets"] == 1
    assert summary["reviewed_packets"] == 1
    assert summary["fully_reviewed_packets"] == 0
