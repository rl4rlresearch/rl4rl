import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture_trajectory_visualization.annotations import enrich_run

from experiments.ontology_categorical_dashboard import METRIC_VIEW_VERSION
from experiments.ontology_categorical_fingerprint import (
    FINGERPRINT_VERSION,
    METRICS,
    campaign_schema,
    schema_revision,
)


class ArchitectureAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "outputs/ontology-categorical-v1"
        self.schema = dict(campaign_schema("fashion"))
        self.schema["data_directory"] = "test-campaign"
        self.schema["proposal_limit"] = 2
        self.revision = schema_revision(self.schema)
        fp = {
            component["id"]: component["values"][0]
            for component in self.schema["components"]
        }
        second = dict(fp)
        component = next(
            item for item in self.schema["components"] if len(item["values"]) > 1
        )
        self.changed = component["id"]
        second[self.changed] = component["values"][1]
        metrics = {
            f"{stage}:{key}": 0
            for stage in ("implemented", "retained")
            for key in METRICS
        }
        self.rows = [
            {
                "proposal": 0,
                "candidate_id": "seed",
                "fingerprint": fp,
                "metrics": metrics,
                "no_change_assumed": False,
            },
            {
                "proposal": 1,
                "candidate_id": "child",
                "fingerprint": second,
                "metrics": {**metrics, "implemented:component_edits_cumulative": 7},
                "no_change_assumed": False,
            },
            {
                "proposal": 2,
                "candidate_id": "unresolved",
                "fingerprint": None,
                "metrics": {**metrics, "implemented:component_edits_cumulative": 7},
                "no_change_assumed": True,
            },
        ]
        self.compact = {
            "campaign": "fashion",
            "fingerprint_version": FINGERPRINT_VERSION,
            "metric_view_version": METRIC_VIEW_VERSION,
            "schema_revision": self.revision,
            "runs": {"run-1": self.rows},
        }
        self.run = {
            "campaign": "test-campaign",
            "run_id": "run-1",
            "condition": "C0",
            "occurrences": [
                {"id": "s", "proposal": 0, "candidate_id": "seed", "parent_ids": []},
                {
                    "id": "c",
                    "proposal": 1,
                    "candidate_id": "child",
                    "parent_ids": ["seed"],
                },
                {
                    "id": "u",
                    "proposal": 2,
                    "candidate_id": "unresolved",
                    "parent_ids": ["child"],
                },
            ],
        }
        self.publish()

    def publish(self):
        files = {}
        for name, doc in [
            ("working-schema.json", self.schema),
            ("trajectory-metrics.json", self.compact),
        ]:
            path = self.base / "forks/fashion" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(doc).encode()
            path.write_bytes(raw)
            files[name] = {
                "path": str(path.relative_to(self.base)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        manifest = {
            "format": "ontology-publication-v1",
            "campaigns": {
                "fashion": {
                    "schema_revision": self.revision,
                    "fingerprint_version": FINGERPRINT_VERSION,
                    "files": files,
                }
            },
        }
        (self.base / "publication-manifest.json").write_text(json.dumps(manifest))
        (self.base / "forks/fashion/final.json").write_text("MUST NOT READ")

    def test_exact_joins_metrics_and_parent_changes_without_mutation(self):
        original = copy.deepcopy(self.run)
        result = enrich_run(self.root, self.run)
        self.assertEqual(self.run, original)
        self.assertTrue(result["categorical_publication"]["checksum_verified"])
        annotation = result["occurrences"][1]["categorical"]
        self.assertEqual(annotation["status"], "reviewed")
        self.assertEqual(annotation["changed_components"], [self.changed])
        self.assertEqual(
            annotation["metrics"]["implemented:component_edits_cumulative"], 7
        )
        self.assertEqual(annotation["source_verification"], "unverified")
        self.assertTrue(annotation["family_id"].startswith("family_"))

    def test_assumptions_do_not_invent_fingerprints(self):
        annotation = enrich_run(self.root, self.run)["occurrences"][2]["categorical"]
        self.assertEqual(annotation["status"], "assumed")
        self.assertIsNone(annotation["fingerprint"])
        self.assertIsNone(annotation["family_id"])
        self.assertIsNone(annotation["changed_components"])
        self.assertEqual(
            annotation["metrics"]["implemented:component_edits_cumulative"], 7
        )

    def test_identity_mismatch_and_scope_are_distinct(self):
        self.run["occurrences"][1]["candidate_id"] = "different"
        self.run["occurrences"].append(
            {"id": "late", "proposal": 3, "candidate_id": "late"}
        )
        result = enrich_run(self.root, self.run)
        self.assertEqual(
            result["occurrences"][1]["categorical"]["status"], "unreviewed"
        )
        self.assertEqual(
            result["occurrences"][-1]["categorical"]["status"], "out_of_scope"
        )
        self.assertIsNone(result["occurrences"][1]["categorical"]["metrics"])

    def test_checksum_failure_hides_reviews(self):
        path = self.base / "forks/fashion/trajectory-metrics.json"
        path.write_bytes(path.read_bytes().replace(b'"child"', b'"other"'))
        result = enrich_run(self.root, self.run)
        self.assertFalse(result["categorical_publication"]["available"])
        self.assertIn("SHA-256", result["categorical_publication"]["reason"])
        self.assertTrue(
            all(
                item["categorical"]["fingerprint"] is None
                for item in result["occurrences"]
            )
        )

    def test_schema_version_failure_never_regenerates_from_final(self):
        self.compact["metric_view_version"] = -1
        self.publish()
        original = Path.read_bytes

        def read(path):
            if path.name == "final.json":
                raise AssertionError("Huge evidence file must not be read")
            return original(path)

        with patch.object(Path, "read_bytes", read):
            result = enrich_run(self.root, self.run)
        self.assertFalse(result["categorical_publication"]["available"])
        self.assertIn("version/schema", result["categorical_publication"]["reason"])

    def test_source_file_digests_when_explicitly_published(self):
        self.rows[1]["source_file_sha256"] = {"model.py": "a" * 64}
        self.publish()
        self.run["occurrences"][1]["source"] = {
            "files": [{"path": "model.py", "sha256": "b" * 64}]
        }
        annotation = enrich_run(self.root, self.run)["occurrences"][1]["categorical"]
        self.assertEqual(annotation["status"], "unreviewed")
        self.assertEqual(annotation["source_verification"], "mismatch")
        self.run["occurrences"][1]["source"]["files"][0]["sha256"] = "a" * 64
        annotation = enrich_run(self.root, self.run)["occurrences"][1]["categorical"]
        self.assertEqual(annotation["source_verification"], "verified")

    def test_unavailable_primary_parent_does_not_choose_secondary(self):
        self.run["occurrences"][1]["parent_ids"] = ["missing", "seed"]
        annotation = enrich_run(self.root, self.run)["occurrences"][1]["categorical"]
        self.assertEqual(annotation["status"], "reviewed")
        self.assertIsNone(annotation["changed_components"])
        self.assertIsNone(annotation["comparison"])


if __name__ == "__main__":
    unittest.main()
