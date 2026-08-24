from pathlib import Path

from rl4rl.adapters.autoresearch import parse_autoresearch_tsv
from rl4rl.schema import EventStatus


def test_parse_autoresearch_tsv(tmp_path: Path) -> None:
    source = tmp_path / "results.tsv"
    source.write_text(
        "commit hash\taccuracy\tparameters\tstatus\tdescription\n"
        "abc\t99.64%\t1,694\tkeep\tbaseline\n"
        "def\t82\t1,680\tdiscard\tshared norm\n",
        encoding="utf-8",
    )
    events = parse_autoresearch_tsv(source, run_id="replication-1")
    assert len(events) == 2
    assert events[0].architecture.accuracy == 0.9964
    assert events[0].architecture.parameters == 1694
    assert events[0].status == EventStatus.ACCEPTED
    assert events[1].status == EventStatus.REJECTED
    assert not events[1].parent_ids


def test_parse_autoresearch_tsv_accepts_blank_measurements_for_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results.tsv"
    source.write_text(
        "commit\taccuracy\tparameters\tstatus\tdescription\n"
        "crash-1\t\t\terror\ttraining crashed before a checkpoint\n"
    )

    events = parse_autoresearch_tsv(source, run_id="replication-1")

    assert events[0].architecture.accuracy is None
    assert events[0].architecture.parameters is None
    assert events[0].status.value == "error"
