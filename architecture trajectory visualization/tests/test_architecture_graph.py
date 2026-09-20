"""Evidence boundaries and graph-diff regression tests; no candidate execution."""

import json
import tempfile
import unittest
from pathlib import Path

from architecture_trajectory_visualization.graph import (
    diff_architectures,
    extract_architecture,
)

MODEL = """
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.embed = nn.Embedding(32, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 32, bias=False)
    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        return self.head(x)
"""


def graph(source=MODEL):
    return extract_architecture({"src/model.py": source})


class ArchitectureGraphTests(unittest.TestCase):
    def test_declared_configuration_and_forward_flow(self):
        result = graph()
        nodes = {node["label"]: node for node in result["nodes"]}
        self.assertEqual(nodes["embed"]["config"]["embedding_dim"], 16)
        self.assertEqual(nodes["head"]["config"]["out_features"], 32)
        edges = {
            (edge["source"], edge["target"])
            for edge in result["edges"]
            if edge["type"] == "data_flow"
        }
        self.assertIn((nodes["embed"]["id"], nodes["norm"]["id"]), edges)
        self.assertIn((nodes["norm"]["id"], nodes["head"]["id"]), edges)
        self.assertTrue(nodes["head"]["is_output"])
        self.assertEqual(result["extraction"]["completeness"], "partial")
        self.assertIsNone(result["summary"]["parameter_count"])

    def test_renaming_does_not_rebuild_structure(self):
        renamed = (
            MODEL.replace("Model", "RenamedModel")
            .replace("embed", "tokens")
            .replace("norm", "normalizer")
            .replace("head", "output")
        )
        before, after = graph(), graph(renamed)
        self.assertEqual(
            [node["id"] for node in before["nodes"]],
            [node["id"] for node in after["nodes"]],
        )
        diff = diff_architectures(before, after)
        self.assertFalse(diff["added"] or diff["removed"] or diff["changed"])
        self.assertFalse(diff["added_edges"] or diff["removed_edges"])

    def test_parameter_change_is_changed_component(self):
        before = graph()
        after = graph(MODEL.replace("nn.Linear(width, 32", "nn.Linear(width, 64"))
        diff = diff_architectures(before, after)
        self.assertEqual(diff["summary"]["changed"], 1)
        self.assertFalse(diff["added"] or diff["removed"])
        self.assertEqual(diff["changed"][0]["after"]["config"]["out_features"], 64)

    def test_sequential_has_order_but_module_list_does_not(self):
        source = """
from torch import nn
class Net(nn.Module):
    def __init__(self):
        self.seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        self.parts = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
"""
        result = graph(source)
        nodes = {node["id"]: node for node in result["nodes"]}
        edges = [edge for edge in result["edges"] if edge["type"] == "data_flow"]
        self.assertEqual(len(edges), 2)
        self.assertTrue(
            all(nodes[edge["source"]]["label"].startswith("seq[") for edge in edges)
        )
        self.assertTrue(any(edge["type"] == "containment" for edge in result["edges"]))

    def test_shared_weights_are_not_dataflow_or_recurrence(self):
        source = MODEL.replace(
            "    def forward",
            "        self.head.weight = self.embed.weight\n    def forward",
        )
        result = graph(source)
        ties = [edge for edge in result["edges"] if edge["type"] == "shared_parameters"]
        self.assertEqual(len(ties), 1)
        self.assertFalse(any(node.get("recurrence") for node in result["nodes"]))
        self.assertEqual(
            diff_architectures(graph(), result)["summary"]["added_edges"], 1
        )

    def test_no_flow_invented_across_control_flow_or_repeated_calls(self):
        source = MODEL.replace(
            "x = self.norm(x)", "for _ in range(3):\n            x = self.norm(x)"
        )
        result = graph(source)
        nodes = {node["label"]: node for node in result["nodes"]}
        self.assertFalse(
            any(edge["target"] == nodes["head"]["id"] for edge in result["edges"])
        )
        self.assertEqual(nodes["norm"]["execution"][0]["kind"], "call_inside_loop")
        repeated = graph(
            MODEL.replace(
                "return self.head(x)", "x = self.norm(x)\n        return self.head(x)"
            )
        )
        norm = next(node for node in repeated["nodes"] if node["label"] == "norm")
        self.assertEqual(len(norm["call_sites"]), 2)
        self.assertFalse(
            any(
                norm["id"] in (edge["source"], edge["target"])
                for edge in repeated["edges"]
            )
        )

    def test_source_is_never_executed_and_parse_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "executed"
            result = graph(f"open({str(target)!r}, 'w').write('bad')\n" + MODEL)
            self.assertFalse(target.exists())
            self.assertTrue(result["nodes"])
        invalid = extract_architecture({"model.py": "def broken(:"})
        self.assertEqual(invalid["extraction"]["completeness"], "unavailable")
        self.assertIn("SyntaxError", " ".join(invalid["extraction"]["warnings"]))

    def test_direct_ir_and_dangling_edges(self):
        ir = {
            "nodes": [
                {"id": "x", "op": "input", "shape": [1, 8]},
                {
                    "id": "dense",
                    "op": "Linear",
                    "inputs": ["x"],
                    "config": {"out_features": 2},
                },
            ]
        }
        result = extract_architecture({}, ir=ir)
        self.assertEqual(result["representation"], "computation_graph")
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["nodes"][0]["shape"], [1, 8])
        bad = extract_architecture(
            {}, ir={**ir, "edges": [{"source": "missing", "target": "dense"}]}
        )
        self.assertEqual(bad["extraction"]["completeness"], "partial")
        self.assertFalse(bad["edges"])
        duplicate = extract_architecture({}, ir={"nodes": [{"id": "x"}, {"id": "x"}]})
        self.assertEqual(duplicate["extraction"]["completeness"], "unavailable")
        invalid_kind = extract_architecture(
            {}, ir={"nodes": [{"id": "x", "kind": ["Linear"]}]}
        )
        self.assertEqual(invalid_kind["extraction"]["completeness"], "unavailable")

    def test_identical_modules_remain_distinct_and_matching_is_marked_ambiguous(self):
        source = MODEL.replace(
            "self.norm = nn.LayerNorm(width)",
            "self.norm = nn.LayerNorm(width)\n        self.norm2 = nn.LayerNorm(width)",
        )
        result = graph(source)
        norms = [node for node in result["nodes"] if node["kind"] == "normalization"]
        self.assertEqual(len(norms), 2)
        self.assertNotEqual(norms[0]["id"], norms[1]["id"])
        self.assertTrue(
            any(
                match["ambiguous"]
                for match in diff_architectures(result, result)["matches"]
            )
        )

    def test_unsafe_numeric_values_stay_json_safe_and_lossless(self):
        result = graph(MODEL.replace("width=16", "width=10000000000000000000"))
        embed = next(node for node in result["nodes"] if node["label"] == "embed")
        self.assertEqual(
            embed["config"]["embedding_dim"], {"integer": "10000000000000000000"}
        )
        huge = extract_architecture(
            {}, ir={"nodes": [{"id": "x", "config": {"value": float("inf")}}]}
        )
        json.dumps(huge, allow_nan=False)
        self.assertEqual(huge["nodes"][0]["config"]["value"]["non_finite"], "inf")

    def test_known_factory_excludes_unused_alternative_classes(self):
        source = (
            MODEL
            + """
class UnusedAlternative(nn.Module):
    def __init__(self):
        self.huge = nn.Linear(10000, 10000)
def build_model():
    return Model()
"""
        )
        result = graph(source)
        self.assertEqual([group["label"] for group in result["groups"]], ["Model"])
        self.assertTrue(result["groups"][0]["is_entrypoint"])
        self.assertFalse(any(node["label"] == "huge" for node in result["nodes"]))

    def test_recorded_architecture_tensor_graph_schema(self):
        ir = {
            "schema_name": "architecture_tensor_graph",
            "schema_version": "1.0",
            "nodes": [
                {
                    "node_id": "tokens",
                    "kind": "token_embedding",
                    "attributes": {},
                    "input_shapes": [["Batch", "Time"]],
                    "output_shape": ["Batch", "Time", 8],
                },
                {
                    "node_id": "head",
                    "kind": "readout",
                    "attributes": {"tie_embedding": "tokens"},
                    "input_shapes": [["Batch", "Time", 8]],
                    "output_shape": ["Batch", "Time", 15],
                },
            ],
            "edges": [
                {"source": "tokens", "target": "head", "kind": "data", "target_port": 0}
            ],
            "output_node_id": "head",
        }
        result = extract_architecture({}, ir=ir)
        self.assertEqual(result["representation"], "computation_graph")
        self.assertEqual(result["nodes"][0]["kind"], "embedding")
        self.assertTrue(result["nodes"][1]["is_output"])
        self.assertEqual(result["nodes"][0]["shape"]["output"], ["Batch", "Time", 8])
        self.assertEqual(
            {edge["type"] for edge in result["edges"]},
            {"data_flow", "shared_parameters"},
        )

    def test_recurrent_step_is_labeled_partial_external_interface(self):
        source = """
from torch import nn
class KeywordGRU(nn.Module):
    def __init__(self):
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128)
    def recurrent_step(self, frame, state):
        output, hidden = self.gru(self.input_norm(frame), state)
        return output, hidden
def build_model():
    return KeywordGRU()
"""
        result = graph(source)
        self.assertTrue(any(node.get("recurrence") for node in result["nodes"]))
        self.assertEqual(result["groups"][0]["interface_method"], "recurrent_step")
        self.assertTrue(
            any(
                "external execution" in warning
                for warning in result["extraction"]["warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
