import json
from pathlib import Path

from experiments import recover_ontology_artifact_sources as recovery


def _write_review(
    root: Path,
    candidate: str,
    source: dict[str, str],
    *,
    reviewer="ok",
    status="validated",
):
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    result = results / f"{candidate}.json"
    result.write_text(
        json.dumps(
            {
                "candidate_id": candidate,
                "status": status,
                "schema_revision": "rev",
                "fingerprint": {"role": "value"},
                "source_review": {"reviewer": reviewer},
            }
        ),
        encoding="utf-8",
    )
    (results / f"{candidate}.source.json").write_text(
        json.dumps({"candidate_id": candidate, "sources": source}),
        encoding="utf-8",
    )


def test_direct_artifact_review_is_preferred_over_digest_fallback(monkeypatch):
    source = {"train.py": "value = 1\n"}
    direct = {
        "path": Path("direct.json"),
        "source": source,
        "template": {"source_review": {}, "fingerprint": {}},
        "validation_attempted": False,
        "validation_receipt": None,
        "validation_error": None,
    }
    fallback = {**direct, "path": Path("fallback.json")}
    monkeypatch.setattr(recovery, "validate_source_review", lambda *_args: {})

    selected = recovery._find_review(
        "artifact-id",
        source,
        {"artifact-id": [direct]},
        {recovery.source_digest(source): [fallback]},
        {},
    )

    assert selected is direct


def test_stale_or_invalid_direct_review_bypasses_to_exact_digest_match(
    tmp_path, monkeypatch
):
    source = {"train.py": "value = 1\n"}
    direct_id = "a" * 64
    fallback_id = "b" * 64
    _write_review(tmp_path, direct_id, source, reviewer="bad")
    _write_review(tmp_path, fallback_id, source)

    calls = []

    def fake_validate(review, *_args):
        calls.append(review["reviewer"])
        if review.get("reviewer") == "bad":
            raise ValueError("invalid")
        return {}

    monkeypatch.setattr(recovery, "validate_source_review", fake_validate)
    by_identity, by_digest = recovery._review_entries(tmp_path, "rev")

    assert direct_id in by_identity
    assert calls == []
    selected = recovery._find_review(direct_id, source, by_identity, by_digest, {})
    assert selected["path"].stem == fallback_id
    assert calls == ["bad", "ok"]
    assert (
        recovery._find_review(direct_id, source, by_identity, by_digest, {}) is selected
    )
    assert calls == ["bad", "ok"]


def test_different_bundle_is_rejected_even_when_identity_is_known():
    recorded = {"train.py": "value = 1\n"}
    reviewed = {"train.py": "value = 2\n"}
    entry = {
        "path": Path("review.json"),
        "source": reviewed,
        "template": {"source_review": {}, "fingerprint": {}},
        "validation_attempted": False,
        "validation_receipt": None,
        "validation_error": None,
    }

    assert (
        recovery._find_review(
            "artifact-id",
            recorded,
            {"artifact-id": [entry]},
            {recovery.source_digest(reviewed): [entry]},
            {},
        )
        is None
    )
