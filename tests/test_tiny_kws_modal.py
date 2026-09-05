import ast
from pathlib import Path

from experiments.c0c3_factorial.spec import TaskSpec
from experiments.c0c3_factorial.tiny_v21_runtime import ModalFallbackEvaluator


def test_kws_modal_resources_and_data_image():
    source = Path("experiments/modal_tiny_kws_app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    options = {
        k.arg: ast.literal_eval(k.value)
        for k in function.decorator_list[0].keywords
        if k.arg in {"cpu", "memory", "retries"}
    }
    assert options == {"cpu": 2, "memory": 4096, "retries": 0}
    assert not any(
        k.arg == "max_containers" for k in function.decorator_list[0].keywords
    )
    assert "/opt/rl4rl/data/raw/tiny-kws-rnn" in source
    assert "/opt/rl4rl/experiments/modal_tiny_kws_app.py" in source


def test_kws_uses_shared_three_slot_fallback(tmp_path):
    task = TaskSpec.from_toml(
        Path("experiments/c0c3_factorial/configs/tasks/tiny_kws_rnn_source_only_cpu.toml")
    )
    evaluator = ModalFallbackEvaluator(
        task=task,
        support_source=tmp_path,
        repo_root=tmp_path,
        python_bin="python",
        options={"modal_app": "rl4rl-tiny-kws-cpu", "local_evaluator_capacity": 3},
    )
    assert evaluator.local_capacity == 3
    assert evaluator.fallback_root == tmp_path / "data/c0c3/.tiny-v21-local-evaluators"
