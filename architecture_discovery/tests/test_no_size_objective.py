import ast
import inspect
import textwrap
from pathlib import Path

import yaml

from evaluation.records import CONTROLLER_SEARCH_FIELDS, SearchEvaluationRecord
from common.trainer import checkpoint_is_better


ROOT = Path(__file__).resolve().parents[1]


def test_controller_search_view_exposes_trusted_parameter_metadata():
    assert "parameter_count_metadata" in CONTROLLER_SEARCH_FIELDS
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(SearchEvaluationRecord.controller_view))
    )
    assert "parameter_count_metadata" in ast.unparse(tree)


def test_checkpoint_ranking_has_no_parameter_count_input():
    signature = inspect.signature(checkpoint_is_better)
    assert all("parameter" not in name for name in signature.parameters)


def test_configurations_enable_accuracy_constrained_parameter_selection():
    greedy = yaml.safe_load(
        (ROOT / "agents" / "greedy_autoresearch" / "config.yaml").read_text()
    )
    semantic = yaml.safe_load(
        (ROOT / "agents" / "semantic_autoresearch" / "config.yaml").read_text()
    )
    assert greedy["acceptance"]["use_parameter_count"] is True
    assert semantic["acceptance"]["use_parameter_count"] is True
