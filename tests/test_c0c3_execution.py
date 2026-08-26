# ruff: noqa: E402 -- the standalone experiments package is added explicitly.

from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import io
import json
import shutil
import stat
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial import task_evaluators
from experiments.c0c3_factorial.artifacts import (
    candidate_hash,
    materialize_candidate,
    snapshot_candidate,
)
from experiments.c0c3_factorial.campaign import (
    calibrate_task,
    create_campaign,
    execute_calibration,
    prepare_calibration,
)
from experiments.c0c3_factorial.codex_cli import CodexCli
from experiments.c0c3_factorial.evaluator import (
    CommandEvaluator,
    make_command_evaluator,
    shared_local_evaluator_status,
)
from experiments.c0c3_factorial.frameworks import (
    OpenEvolveAdapter,
    bundle_workspace,
    parse_metadata,
    unbundle_workspace,
)
from experiments.c0c3_factorial.hybrid_evaluator import (
    NANOGPT_APP_NAME,
    ModalCommandEvaluator,
    _archive_inputs,
    _extract_outputs,
    _remote_target,
)
from experiments.c0c3_factorial.modal_app import safe_campaign_path
from experiments.c0c3_factorial.neutral_task import (
    AUTORESEARCH_V17_PROMPT_PROFILE,
    NANOGPT_TASK_ADAPTER,
    NEUTRAL_PROMPT_PROFILE,
    NEUTRAL_TASK_ADAPTER,
    OPENEVOLVE_V2_PROMPT_PROFILE,
    PAIR_TOKEN_TASK_ADAPTER_V2,
    PAIR_TOKEN_TASK_ADAPTER_V3,
)
from experiments.c0c3_factorial.orchestration import (
    FACTORIAL_STAGE,
    NO_SEARCH_STAGE,
    CampaignLockedError,
    IndependentTrajectoryError,
    ParallelWaveError,
    campaign_lock,
    next_parallel_wave,
    next_run,
    next_staged_parallel_wave,
    request_staged_trajectory_pause,
    run_parallel_campaign,
    run_staged_campaign,
    run_staged_independent_campaign,
    run_staged_individual_trajectory,
    staged_independent_trajectories,
)
from experiments.c0c3_factorial.postsearch import (
    export_layer_b_packets,
    run_layer_c,
    score_layer_b,
)
from experiments.c0c3_factorial.prompts import RenderedPrompt
from experiments.c0c3_factorial.runner import (
    recover_active_opportunity,
    run_one_opportunity,
)
from experiments.c0c3_factorial.spec import (
    OPENEVOLVE_V2_EXECUTION_RULE,
    PARALLEL_EXECUTION_RULE,
    STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
    STAGED_INDEPENDENT_EXECUTION_RULE,
    STAGED_INDIVIDUAL_EXECUTION_RULE,
    STAGED_PARALLEL_EXECUTION_RULE,
    BudgetSpec,
    ConversationMode,
    ExecutionBackend,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    ModelSpec,
    ObjectiveDirection,
    TaskSpec,
)
from experiments.c0c3_factorial.state import Evaluation, SearchController, Usage
from experiments.c0c3_factorial.validation import validate_campaign


def test_step_zero_best_uses_distinct_trained_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(
        {"step": 0, "model_state": {"weight": torch.tensor([0.0])}},
        checkpoints / "best.pt",
    )
    torch.save(
        {"step": 4999, "model_state": {"weight": torch.tensor([1.0])}},
        checkpoints / "last.pt",
    )

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4))

        def forward(self, inputs):
            return inputs @ self.weight

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = FakeAttention()

        def forward(self, tokens):
            encoded = torch.nn.functional.one_hot(tokens % 4, num_classes=4)
            return self.attention(encoded.float())

    submission = SimpleNamespace(
        BOS_ID=0,
        preprocess=lambda _a, _b: [1],
        encode=lambda values: values,
        build_model=lambda: (FakeModel(), {}),
        add=lambda _model, _a, _b: -1,
    )
    monkeypatch.setattr(task_evaluators, "_load_submission", lambda _path: submission)

    assert task_evaluators._trained_model_contract_error(tmp_path) is None


def test_step_zero_best_without_trained_final_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(
        {"step": 0, "model_state": {"weight": torch.tensor([0.0])}},
        checkpoints / "best.pt",
    )

    assert task_evaluators._trained_model_contract_error(tmp_path) == (
        "the saved model does not record a positive training step"
    )


def protocol() -> FactorialSpec:
    return FactorialSpec(
        protocol_version="1.0",
        study_id="execution-test",
        study_seed=42,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(1,),
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(
            proposals=1,
            candidate_evaluations=1,
            max_total_tokens=1000,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
    )


def parallel_protocol() -> FactorialSpec:
    serial = protocol()
    return FactorialSpec(
        **{
            **serial.__dict__,
            "protocol_version": "1.1",
            "study_id": "parallel-execution-test",
            "execution_rule": PARALLEL_EXECUTION_RULE,
        }
    )


def staged_protocol() -> FactorialSpec:
    base = protocol()
    return FactorialSpec(
        **{
            **base.__dict__,
            "protocol_version": "1.3",
            "study_id": "staged-execution-test",
            "blocks": 2,
            "conversation_mode": ConversationMode.CONTINUOUS,
            "execution_rule": STAGED_PARALLEL_EXECUTION_RULE,
        }
    )


def independent_staged_protocol() -> FactorialSpec:
    base = staged_protocol()
    return FactorialSpec(
        **{
            **base.__dict__,
            "protocol_version": "1.4",
            "study_id": "independent-staged-execution-test",
            "budget": BudgetSpec(
                proposals=2,
                candidate_evaluations=2,
                max_total_tokens=1000,
                max_evaluator_seconds=100.0,
                evaluator_timeout_seconds=10,
            ),
            "transition_opportunities": (2,),
            "execution_rule": STAGED_INDEPENDENT_EXECUTION_RULE,
        }
    )


def individually_controlled_staged_protocol() -> FactorialSpec:
    base = independent_staged_protocol()
    return FactorialSpec(
        **{
            **base.__dict__,
            "protocol_version": "1.5",
            "study_id": "individual-staged-execution-test",
            "execution_rule": STAGED_INDIVIDUAL_EXECUTION_RULE,
        }
    )


def task(seed_source: Path) -> TaskSpec:
    return TaskSpec(
        task_id="toy_execution",
        display_name="Toy execution task",
        adapter="command_json_v1",
        seed_source=str(seed_source),
        editable_paths=("candidate.py",),
        evaluator_command=(
            "{python}",
            "evaluate.py",
            "--output",
            "{output}",
        ),
        objective_metric="score",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        qualification_metric=None,
        qualification_minimum=None,
        public_feedback_metrics=("score",),
        metric_patterns={"score": r"score:\s*([0-9.]+)"},
        final_holdout_command=(
            "{python}",
            "evaluate.py",
            "--output",
            "{output}",
        ),
        preferred_backend=ExecutionBackend.LOCAL,
    )


def framework() -> FrameworkSpec:
    return FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_v1",
        prompt_profile="controlled_factorial_v1",
        edit_mode="direct_workspace",
    )


def make_seed(root: Path) -> Path:
    root.mkdir()
    (root / "candidate.py").write_text("SCORE = 0\n", encoding="utf-8")
    (root / "evaluate.py").write_text(
        """\
import argparse
import json
from candidate import SCORE

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.output, 'w') as handle:
    json.dump({'metrics': {'score': SCORE}}, handle)
print(f'score: {SCORE}')
""",
        encoding="utf-8",
    )
    return root


def make_neutral_seed(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "src" / "model.py").write_text("SCORE = 0\n", encoding="utf-8")
    (root / "src" / "data.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "checkpoints" / "best.pt").write_text("seed", encoding="utf-8")
    return root


def make_pair_token_seed(root: Path) -> Path:
    for relative, source in {
        "src/__init__.py": "",
        "src/model.py": (
            "class ModelConfig:\n    pass\nclass TinyDecoderLM:\n    pass\n"
        ),
        "src/data.py": (
            "BOS_ID = 0\n"
            "def preprocess(a, b):\n    return [a, b]\n"
            "def postprocess(value):\n    return value\n"
        ),
        "src/eval.py": "VALUE = 1\n",
        "src/train.py": "VALUE = 1\n",
        "checkpoints/best.pt": "seed-checkpoint",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def openevolve_v2_protocol(*, blocks: int = 3) -> FactorialSpec:
    return FactorialSpec(
        protocol_version="2.0",
        study_id="openevolve-v2-execution-test",
        study_seed=20260823,
        blocks=blocks,
        portfolio_capacity=4,
        transition_opportunities=(1,),
        conversation_mode=ConversationMode.EPHEMERAL,
        model=ModelSpec("gpt-fake", "xhigh"),
        budget=BudgetSpec(
            proposals=1,
            candidate_evaluations=1,
            max_total_tokens=1000,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
        execution_rule=OPENEVOLVE_V2_EXECUTION_RULE,
        include_no_search=False,
    )


def openevolve_v2_task(seed_source: Path) -> TaskSpec:
    command = ("{python}", "-c", "raise SystemExit(0)")
    return TaskSpec(
        task_id="pair-transformer-v2-test",
        display_name="trained transformer for 10-digit addition",
        adapter=PAIR_TOKEN_TASK_ADAPTER_V2,
        seed_source=str(seed_source),
        editable_paths=("src/model.py", "src/train.py"),
        evaluator_command=command,
        objective_metric="parameters",
        objective_direction=ObjectiveDirection.MINIMIZE,
        qualification_metric="accuracy",
        qualification_minimum=0.99,
        public_feedback_metrics=("accuracy", "parameters", "training_steps"),
        metric_patterns={},
        final_holdout_command=command,
        preferred_backend=ExecutionBackend.LOCAL,
    )


def openevolve_v2_framework() -> FrameworkSpec:
    return FrameworkSpec(
        framework_id=FrameworkKind.OPENEVOLVE,
        adapter="controlled_openevolve_prompt_diff_v2",
        prompt_profile=OPENEVOLVE_V2_PROMPT_PROFILE,
        edit_mode="search_replace_diff",
    )


def openevolve_v21_task(seed_source: Path) -> TaskSpec:
    base = openevolve_v2_task(seed_source)
    return TaskSpec(**{**base.__dict__, "adapter": PAIR_TOKEN_TASK_ADAPTER_V3})


def artifact_clean_task(seed_source: Path) -> TaskSpec:
    command = (
        "{python}",
        "-c",
        (
            "import json,sys; "
            "open(sys.argv[1], 'w').write(json.dumps("
            "{{'metrics': "
            "{{'accuracy': 1.0, 'parameters': 1644, 'training_steps': 1, "
            "'cases': 10010, 'correct': 10010}}}}))"
        ),
        "{output}",
    )
    return TaskSpec(
        task_id="artifact-clean-pair-transformer",
        display_name="transformer for 10-digit addition",
        adapter=PAIR_TOKEN_TASK_ADAPTER_V3,
        seed_source=str(seed_source),
        editable_paths=("src/model.py", "src/train.py"),
        evaluator_command=command,
        objective_metric="parameters",
        objective_direction=ObjectiveDirection.MINIMIZE,
        qualification_metric="accuracy",
        qualification_minimum=0.99,
        public_feedback_metrics=("accuracy", "parameters", "training_steps"),
        metric_patterns={},
        final_holdout_command=command,
        preferred_backend=ExecutionBackend.LOCAL,
    )


def neutral_task(seed_source: Path) -> TaskSpec:
    command = (
        "{python}",
        "-c",
        (
            "import json,sys; "
            "open(sys.argv[1], 'w').write(json.dumps("
            "{{'metrics': {{'parameters': 1, 'accuracy': 1.0}}}}))"
        ),
        "{output}",
    )
    return TaskSpec(
        task_id="neutral-toy-execution",
        display_name="trained transformer for 10-digit addition",
        adapter=NEUTRAL_TASK_ADAPTER,
        seed_source=str(seed_source),
        editable_paths=("src/model.py", "src/data.py", "src/train.py"),
        evaluator_command=command,
        objective_metric="parameters",
        objective_direction=ObjectiveDirection.MINIMIZE,
        qualification_metric="accuracy",
        qualification_minimum=0.99,
        public_feedback_metrics=("accuracy", "parameters"),
        metric_patterns={},
        final_holdout_command=command,
        preferred_backend=ExecutionBackend.LOCAL,
    )


def make_fake_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
assert '-a' not in args
assert 'approval_policy="never"' in args
workspace = pathlib.Path(args[args.index('--cd') + 1])
last = pathlib.Path(args[args.index('--output-last-message') + 1])
_prompt = sys.stdin.read()
(workspace / 'candidate.py').write_text('SCORE = 1\\n')
last.write_text(
    'HYPOTHESIS: increasing the toy score improves fitness\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
)
print(json.dumps({{'type': 'thread.started', 'thread_id': 'fake'}}))
print(json.dumps({{
    'type': 'turn.completed',
    'usage': {{
        'input_tokens': 11,
        'cached_input_tokens': 3,
        'output_tokens': 5,
        'reasoning_output_tokens': 2,
    }},
}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_continuous_fake_codex(path: Path) -> Path:
    """Emulate a persisted Codex thread whose cwd is fixed on its first turn."""

    session_workspace = path.with_suffix(".session-workspace")
    invocation_log = path.with_suffix(".invocations.jsonl")
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
session_workspace = pathlib.Path({str(session_workspace)!r})
invocation_log = pathlib.Path({str(invocation_log)!r})
last = pathlib.Path(args[args.index('--output-last-message') + 1])
if 'resume' in args:
    assert '--cd' not in args
    workspace = pathlib.Path(session_workspace.read_text())
    assert pathlib.Path.cwd() == workspace
    mode = 'resume'
else:
    assert '--ephemeral' not in args
    workspace = pathlib.Path(args[args.index('--cd') + 1])
    assert pathlib.Path.cwd() == workspace
    session_workspace.write_text(str(workspace))
    mode = 'initial'
_prompt = sys.stdin.read()
previous = int((workspace / 'candidate.py').read_text().split('=')[1])
(workspace / 'candidate.py').write_text(f'SCORE = {{previous + 1}}\\n')
last.write_text(
    'HYPOTHESIS: increment score\\n'
    'INTENDED_EDIT: increment SCORE\\n'
)
with invocation_log.open('a') as handle:
    handle.write(json.dumps({{'mode': mode, 'args': args}}) + '\\n')
print(json.dumps({{'type': 'thread.started', 'thread_id': 'fake-continuous'}}))
print(json.dumps({{
    'type': 'turn.completed',
    'usage': {{
        'input_tokens': 11,
        'cached_input_tokens': 3,
        'output_tokens': 5,
        'reasoning_output_tokens': 2,
    }},
}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_slow_neutral_continuous_fake_codex(path: Path) -> Path:
    """Edit a sanitized source tree and leave time for a cooperative pause."""

    session_workspace = path.with_suffix(".session-workspace")
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys
import time

args = sys.argv[1:]
session_workspace = pathlib.Path({str(session_workspace)!r})
last = pathlib.Path(args[args.index('--output-last-message') + 1])
if 'resume' in args:
    assert '--cd' not in args
    workspace = pathlib.Path(session_workspace.read_text())
else:
    assert '--ephemeral' not in args
    workspace = pathlib.Path(args[args.index('--cd') + 1])
    session_workspace.write_text(str(workspace))
time.sleep(0.20)
model = workspace / 'src' / 'model.py'
previous = int(model.read_text().split('=')[1])
model.write_text(f'SCORE = {{previous + 1}}\\n')
last.write_text('HYPOTHESIS: increment score\\nINTENDED_EDIT: increment SCORE\\n')
print(json.dumps({{'type': 'thread.started', 'thread_id': 'neutral-continuous'}}))
print(json.dumps({{'type': 'turn.completed', 'usage': {{
    'input_tokens': 11, 'cached_input_tokens': 3,
    'output_tokens': 5, 'reasoning_output_tokens': 2
}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_parallel_barrier_fake_codex(path: Path, marker_root: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys
import time

args = sys.argv[1:]
workspace = pathlib.Path(args[args.index('--cd') + 1])
last = pathlib.Path(args[args.index('--output-last-message') + 1])
run_id = (
    workspace.parent.name
    if workspace.name == '.continuous-codex-workspace'
    else workspace.parents[2].name
)
_prompt = sys.stdin.read()
if not run_id.endswith('-n0'):
    markers = pathlib.Path({str(marker_root)!r})
    markers.mkdir(parents=True, exist_ok=True)
    (markers / run_id).write_text('started')
    deadline = time.monotonic() + 5
    while len(list(markers.iterdir())) < 4:
        if time.monotonic() >= deadline:
            raise SystemExit('four factorial calls did not overlap')
        time.sleep(0.01)
(workspace / 'candidate.py').write_text('SCORE = 1\\n')
last.write_text(
    'HYPOTHESIS: increasing the toy score improves fitness\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
)
print(json.dumps({{'type': 'thread.started', 'thread_id': run_id}}))
print(json.dumps({{
    'type': 'turn.completed',
    'usage': {{
        'input_tokens': 11,
        'cached_input_tokens': 3,
        'output_tokens': 5,
        'reasoning_output_tokens': 2,
    }},
}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_independent_continuous_fake_codex(path: Path, marker_root: Path) -> Path:
    """Require one peer to resume before a deliberately slow peer finishes."""

    session_root = path.with_suffix(".sessions")
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys
import time

args = sys.argv[1:]
markers = pathlib.Path({str(marker_root)!r})
sessions = pathlib.Path({str(session_root)!r})
last = pathlib.Path(args[args.index('--output-last-message') + 1])
if 'resume' in args:
    run_id = last.parents[3].name
    workspace = pathlib.Path((sessions / run_id).read_text())
    (markers / 'resumed').mkdir(parents=True, exist_ok=True)
    (markers / 'resumed' / run_id).write_text('resumed')
else:
    workspace = pathlib.Path(args[args.index('--cd') + 1])
    run_id = workspace.parent.name
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / run_id).write_text(str(workspace))
    initial = markers / 'initial'
    initial.mkdir(parents=True, exist_ok=True)
    (initial / run_id).write_text('started')
    deadline = time.monotonic() + 5
    while len(list(initial.iterdir())) < 4:
        if time.monotonic() >= deadline:
            raise SystemExit('four independent trajectories did not overlap at launch')
        time.sleep(0.01)
    if run_id.endswith('-c0'):
        time.sleep(0.5)
        if not (markers / 'resumed').exists():
            raise SystemExit('C0 waited for its peers before starting opportunity 2')
previous = int((workspace / 'candidate.py').read_text().split('=')[1])
(workspace / 'candidate.py').write_text(f'SCORE = {{previous + 1}}\\n')
last.write_text('HYPOTHESIS: increment score\\nINTENDED_EDIT: increment SCORE\\n')
print(json.dumps({{'type': 'thread.started', 'thread_id': 'fake-' + run_id}}))
print(json.dumps({{'type': 'turn.completed', 'usage': {{
    'input_tokens': 11, 'cached_input_tokens': 3,
    'output_tokens': 5, 'reasoning_output_tokens': 2
}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_failing_fake_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import sys

print('simulated provider transport failure', file=sys.stderr)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_fake_diff_codex(path: Path) -> Path:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
last = pathlib.Path(args[args.index('--output-last-message') + 1])
_prompt = sys.stdin.read()
last.write_text(
    'HYPOTHESIS: a larger score improves the toy objective\\n'
    'INTENDED_EDIT: set SCORE to 1\\n'
    '<<<<<<< SEARCH\\nSCORE = 0\\n=======\\nSCORE = 1\\n>>>>>>> REPLACE\\n'
)
print(json.dumps({{'type': 'turn.completed', 'usage': {{
    'input_tokens': 8, 'cached_input_tokens': 2, 'output_tokens': 4,
    'reasoning_output_tokens': 1
}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_fake_v2_diff_codex(path: Path) -> Path:
    prompt_log = path.with_suffix(".prompt.md")
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

args = sys.argv[1:]
last = pathlib.Path(args[args.index('--output-last-message') + 1])
prompt = sys.stdin.read()
pathlib.Path({str(prompt_log)!r}).write_text(prompt)
last.write_text(
    'MECHANISM: change the training constant\\n'
    'HYPOTHESIS: changing the constant exercises strict patching\\n'
    'INTENDED_EDIT: set the training constant to two\\n'
    'EVIDENCE: the supplied parent contains the old constant\\n'
    '<<<<<<< SEARCH\\nVALUE = 1\\n=======\\nVALUE = 2\\n>>>>>>> REPLACE\\n'
)
print(json.dumps({{'type': 'turn.completed', 'usage': {{
    'input_tokens': 8, 'cached_input_tokens': 2, 'output_tokens': 4,
    'reasoning_output_tokens': 1
}}}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_candidate_snapshot_and_bundle_round_trip(tmp_path: Path) -> None:
    workspace = make_seed(tmp_path / "seed")
    identifier, snapshot = snapshot_candidate(
        workspace, tmp_path / "candidates", ("candidate.py",)
    )
    assert identifier == candidate_hash(workspace, ("candidate.py",))
    materialized = tmp_path / "materialized"
    materialize_candidate(workspace, snapshot, materialized, ("candidate.py",))
    bundle = bundle_workspace(materialized, ("candidate.py",))
    (materialized / "candidate.py").write_text("broken", encoding="utf-8")
    unbundle_workspace(bundle, materialized, ("candidate.py",))
    assert (materialized / "candidate.py").read_text() == "SCORE = 0\n"


def test_metadata_parser() -> None:
    hypothesis, edit = parse_metadata(
        "HYPOTHESIS: use a different family\nINTENDED_EDIT: replace attention\n"
    )
    assert hypothesis == "use a different family"
    assert edit == "replace attention"


def test_modal_campaign_path_is_relative_and_confined() -> None:
    assert str(safe_campaign_path("paper/adderboard")) == (
        "/campaigns/paper/adderboard"
    )
    for unsafe in ("/absolute", "../escape", "a/../../escape", "."):
        try:
            safe_campaign_path(unsafe)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"unsafe Modal path accepted: {unsafe}")


def test_campaign_writer_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with (
        campaign_lock(campaign),
        pytest.raises(CampaignLockedError, match="active writer"),
        campaign_lock(campaign),
    ):
        pass


def test_evaluator_preserves_symlinked_virtualenv_interpreter(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    environment_bin = tmp_path / "environment" / "bin"
    environment_bin.mkdir(parents=True)
    environment_python = environment_bin / "python"
    environment_python.symlink_to(Path(sys.executable).resolve())

    evaluator = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=str(environment_python),
    )

    assert evaluator.python_bin == str(environment_python.absolute())
    assert Path(evaluator.python_bin).is_symlink()
    assert Path(evaluator.python_bin).resolve() == Path(sys.executable).resolve()


def test_evaluator_honors_structured_invalid_result_and_failure_kind(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    (seed_source / "evaluate.py").write_text(
        """\
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.output, 'w') as handle:
    json.dump({
        'valid': False,
        'failure_kind': 'fidelity_screen_not_promoted',
        'metrics': {'score': 0.75},
    }, handle)
""",
        encoding="utf-8",
    )
    _candidate_id, snapshot = snapshot_candidate(
        seed_source, tmp_path / "candidates", ("candidate.py",)
    )
    artifacts = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=sys.executable,
    ).evaluate(
        candidate_snapshot=snapshot,
        opportunity_root=tmp_path / "opportunity",
        timeout_seconds=2,
    )

    assert artifacts.evaluation.valid is False
    assert artifacts.evaluation.fitness is None
    assert artifacts.evaluation.failure_kind == "fidelity_screen_not_promoted"
    assert artifacts.evaluation.metrics["score"] == 0.75


def test_evaluator_slot_serializes_trainers_without_charging_queue_time(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    evaluator_source = seed_source / "evaluate.py"
    evaluator_source.write_text(
        "import time\ntime.sleep(0.2)\n" + evaluator_source.read_text(),
        encoding="utf-8",
    )
    candidate_id, snapshot = snapshot_candidate(
        seed_source, tmp_path / "candidates", ("candidate.py",)
    )
    assert candidate_id
    evaluator = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=sys.executable,
        slot_root=tmp_path / "slots",
        max_parallel_evaluators=1,
    )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                evaluator.evaluate,
                candidate_snapshot=snapshot,
                opportunity_root=tmp_path / f"opportunity-{index}",
                timeout_seconds=2,
            )
            for index in range(2)
        ]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - started

    assert elapsed >= 0.35
    assert all(result.evaluation.valid for result in results)
    assert all(result.evaluation.evaluator_seconds < 0.5 for result in results)
    assert all(
        (tmp_path / f"opportunity-{index}" / "evaluator-queue.json").is_file()
        for index in range(2)
    )


def test_shared_evaluator_slots_serialize_distinct_campaigns(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    evaluator_source = seed_source / "evaluate.py"
    evaluator_source.write_text(
        "import time\ntime.sleep(0.2)\n" + evaluator_source.read_text(),
        encoding="utf-8",
    )
    _candidate_id, snapshot = snapshot_candidate(
        seed_source, tmp_path / "candidates", ("candidate.py",)
    )
    shared_root = tmp_path / "shared-slots"
    evaluators = [
        CommandEvaluator(
            task=task(seed_source),
            support_source=seed_source,
            repo_root=ROOT,
            python_bin=sys.executable,
            slot_root=tmp_path / f"campaign-{index}-slots",
            max_parallel_evaluators=2,
            shared_slot_root=shared_root,
            max_shared_parallel_evaluators=1,
        )
        for index in range(2)
    ]

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                evaluator.evaluate,
                candidate_snapshot=snapshot,
                opportunity_root=tmp_path / f"shared-opportunity-{index}",
                timeout_seconds=2,
            )
            for index, evaluator in enumerate(evaluators)
        ]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - started

    assert elapsed >= 0.35
    assert all(result.evaluation.valid for result in results)
    queues = [
        json.loads(
            (tmp_path / f"shared-opportunity-{index}" / "evaluator-queue.json")
            .read_text(encoding="utf-8")
        )
        for index in range(2)
    ]
    assert all(queue["schema_version"] == "2.0" for queue in queues)
    assert all(queue["shared_capacity"] == 1 for queue in queues)
    assert all(Path(queue["shared_slot"]).parent == shared_root for queue in queues)


def test_shared_evaluator_status_and_remote_bypass(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    shared_root = tmp_path / "shared-slots"
    local = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=sys.executable,
        slot_root=tmp_path / "local-campaign-slots",
        max_parallel_evaluators=1,
        shared_slot_root=shared_root,
        max_shared_parallel_evaluators=1,
    )
    remote = CommandEvaluator(
        task=task(seed_source),
        support_source=seed_source,
        repo_root=ROOT,
        python_bin=sys.executable,
        slot_root=tmp_path / "remote-campaign-slots",
        max_parallel_evaluators=1,
        shared_slot_root=shared_root,
        max_shared_parallel_evaluators=1,
    )
    local_opportunity = tmp_path / "local-opportunity"
    remote_opportunity = tmp_path / "remote-opportunity"
    local_opportunity.mkdir()
    remote_opportunity.mkdir()

    with local._evaluation_slot(local_opportunity):
        status = shared_local_evaluator_status(shared_root, capacity=1)
        assert status["occupied"] == 1
        assert status["available"] == 0
        assert status["slots"][0]["holder"]["opportunity_root"] == str(
            local_opportunity
        )

        started = time.monotonic()
        with remote._evaluation_slot(
            remote_opportunity, include_shared_local_pool=False
        ):
            pass
        assert time.monotonic() - started < 0.1

    assert shared_local_evaluator_status(shared_root, capacity=1)["occupied"] == 0


def test_codex_transport_campaign_and_one_real_controller_step(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_fake_codex(tmp_path / "fake-codex")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is True
    schedule = json.loads((campaign / "schedule.json").read_text())
    c0 = next(row for row in schedule if row["condition"] == "C0")
    record = run_one_opportunity(
        campaign / "runs" / c0["run_id"],
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
        python_bin=sys.executable,
        codex_binary=str(fake_codex),
        codex_timeout_seconds=10,
    )
    assert record["retained"] is True
    assert record["evaluation"]["metrics"]["score"] == 1
    assert record["usage_increment"]["total_tokens"] == 16
    controller = SearchController.load(campaign / "runs" / c0["run_id"], protocol())
    assert controller.state.status == "completed"
    assert controller.state.proposals_used == 1
    assert controller.state.evaluations_used == 1
    n0 = next(row for row in schedule if row["condition"] == "N0")
    n0_state = json.loads((campaign / "runs" / n0["run_id"] / "state.json").read_text())
    assert n0_state["no_search"] is True
    assert n0_state["condition"] == "N0"


def test_continuous_autoresearch_resumes_one_codex_session_per_run(
    tmp_path: Path,
) -> None:
    spec = FactorialSpec(
        protocol_version="1.2",
        study_id="continuous-execution-test",
        study_seed=42,
        blocks=1,
        portfolio_capacity=2,
        transition_opportunities=(2,),
        conversation_mode=ConversationMode.CONTINUOUS,
        model=ModelSpec("gpt-fake", "high"),
        budget=BudgetSpec(
            proposals=2,
            candidate_evaluations=2,
            max_total_tokens=1000,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
        execution_rule=PARALLEL_EXECUTION_RULE,
    )
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_continuous_fake_codex(tmp_path / "fake-continuous-codex")
    continuous_framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_session_resume_v1",
        prompt_profile="controlled_factorial_continuous_v1",
        edit_mode="direct_workspace",
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=continuous_framework,
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    c0 = next(
        row
        for row in json.loads((campaign / "schedule.json").read_text())
        if row["condition"] == "C0"
    )
    run_dir = campaign / "runs" / c0["run_id"]
    for _ in range(2):
        run_one_opportunity(
            run_dir,
            spec=spec,
            task=task(seed_source),
            framework=continuous_framework,
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
        )
    state = SearchController.load(run_dir, spec).state
    assert state.conversation_session_id == "fake-continuous"
    assert state.incumbent_id in state.candidates
    assert state.candidates[state.incumbent_id].metrics["score"] == 2
    invocations = [
        json.loads(line)
        for line in fake_codex.with_suffix(".invocations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["mode"] for entry in invocations] == ["initial", "resume"]
    for entry in invocations:
        assert "--ignore-user-config" in entry["args"]
        assert "--ignore-rules" in entry["args"]
        assert "--strict-config" in entry["args"]
        assert "sandbox_workspace_write.network_access=false" in entry["args"]


def test_portable_calibration_can_execute_on_a_later_backend(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    calibration = prepare_calibration(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
    )
    assert not (calibration / "baseline.json").exists()
    baseline = execute_calibration(
        calibration,
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    assert json.loads(baseline.read_text())["calibration_kind"] == (
        "executed_on_target_backend"
    )


def test_openevolve_v2_campaign_is_twelve_primary_c0c3_runs_without_n0(
    tmp_path: Path,
) -> None:
    spec = openevolve_v2_protocol()
    seed_source = make_pair_token_seed(tmp_path / "source")
    task_spec = openevolve_v2_task(seed_source)
    framework_spec = openevolve_v2_framework()
    calibration = prepare_calibration(
        tmp_path / "calibration",
        spec=spec,
        task=task_spec,
        repo_root=ROOT,
    )
    prepared = json.loads((calibration / "calibration.json").read_text())
    baseline = {
        "schema_version": "1.0",
        "task_id": task_spec.task_id,
        "candidate_id": prepared["candidate_id"],
        "support_tree_sha256": prepared["support_tree_sha256"],
        "fitness": -1644.0,
        "metrics": {
            "accuracy": 1.0,
            "parameters": 1644,
            "training_steps": 5000,
        },
        "evaluator_seconds": 0.0,
        "protocol_hash": spec.protocol_hash,
        "calibration_kind": "executed_on_target_backend",
    }
    baseline_path = calibration / "baseline.json"
    baseline_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task_spec,
        framework=framework_spec,
        calibration_path=baseline_path,
        repo_root=ROOT,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    manifest = json.loads((campaign / "campaign.json").read_text())

    assert len(schedule) == 12
    assert {row["condition"] for row in schedule} == {"C0", "C1", "C2", "C3"}
    assert all(
        sorted(row["condition"] for row in schedule if row["block"] == block)
        == ["C0", "C1", "C2", "C3"]
        for block in (1, 2, 3)
    )
    assert manifest["include_no_search"] is False
    assert manifest["primary_run_ids"] == [row["run_id"] for row in schedule]
    assert manifest["optional_run_ids"] == []
    assert validate_campaign(
        campaign,
        spec=spec,
        task=task_spec,
        framework=framework_spec,
        repo_root=ROOT,
    )["valid"] is True

    with pytest.raises(ValueError, match="forbids N0"):
        create_campaign(
            tmp_path / "invalid-campaign",
            spec=spec,
            task=task_spec,
            framework=framework_spec,
            calibration_path=baseline_path,
            repo_root=ROOT,
            include_no_search=True,
        )


def test_hybrid_modal_transport_archives_only_explicit_inputs_and_is_fail_closed(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support"
    candidate = tmp_path / "candidate"
    support.mkdir()
    candidate.mkdir()
    (support / "protected.py").write_text("VALUE = 1\n", encoding="utf-8")
    (candidate / "model.py").write_text("VALUE = 2\n", encoding="utf-8")

    payload = _archive_inputs(support, candidate)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "candidate/model.py",
            "support/protected.py",
        }

    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe archive path"):
        _extract_outputs(unsafe.getvalue(), tmp_path / "outputs")
    assert not (tmp_path / "outside.txt").exists()


def test_hybrid_modal_backend_uses_evaluator_only_transport(tmp_path: Path) -> None:
    task_spec = openevolve_v2_task(make_pair_token_seed(tmp_path / "source"))
    hybrid_task = TaskSpec(
        **{**task_spec.__dict__, "preferred_backend": ExecutionBackend.HYBRID_MODAL}
    )

    evaluator = make_command_evaluator(
        task=hybrid_task,
        support_source=tmp_path / "support",
        repo_root=ROOT,
        python_bin=sys.executable,
    )

    assert isinstance(evaluator, ModalCommandEvaluator)


def test_nanogpt_hybrid_transport_uses_dedicated_h100_service() -> None:
    app_name, function_name = _remote_target(NANOGPT_TASK_ADAPTER)

    assert app_name == NANOGPT_APP_NAME
    assert function_name == "evaluate_candidate"


def test_hybrid_usage_receipts_support_calibration_and_search_paths(
    tmp_path: Path,
) -> None:
    calibration_evaluation = tmp_path / "calibration" / "evaluation"
    calibration_evaluation.mkdir(parents=True)
    ModalCommandEvaluator._record_usage(
        calibration_evaluation,
        call_id="calibration-call",
        local_wall_seconds=12.0,
        worker_seconds=11.0,
        gpu_name="H100",
        status="completed",
        app_name=NANOGPT_APP_NAME,
        function_name="evaluate_candidate",
    )
    calibration_record = json.loads(
        (calibration_evaluation / "modal-usage.json").read_text()
    )
    assert calibration_record["record_kind"] == "baseline_calibration"
    assert calibration_record["run_id"] == "[calibration]"
    assert calibration_record["opportunity"] == 0
    assert (tmp_path / "calibration/modal-usage.jsonl").is_file()

    search_evaluation = (
        tmp_path / "campaign/runs/example/opportunities/0007"
    )
    search_evaluation.mkdir(parents=True)
    ModalCommandEvaluator._record_usage(
        search_evaluation,
        call_id="search-call",
        local_wall_seconds=22.0,
        worker_seconds=21.0,
        gpu_name="H100",
        status="completed",
        app_name=NANOGPT_APP_NAME,
        function_name="evaluate_candidate",
    )
    search_record = json.loads(
        (search_evaluation / "modal-usage.json").read_text()
    )
    assert search_record["record_kind"] == "candidate_evaluation"
    assert search_record["run_id"] == "example"
    assert search_record["opportunity"] == 7
    assert (tmp_path / "campaign/modal-usage.jsonl").is_file()


def test_nanogpt_calibration_round_trip_accepts_modal_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_source = tmp_path / "autoresearch"
    seed_source.mkdir()
    (seed_source / "prepare.py").write_text("# fixed evaluator utility\n")
    (seed_source / "train.py").write_text("# editable research program\n")
    task_spec = TaskSpec(
        task_id="nanogpt-calibration-test",
        display_name="fixed-time language-model pretraining",
        adapter=NANOGPT_TASK_ADAPTER,
        seed_source=str(seed_source),
        editable_paths=("train.py",),
        evaluator_command=("{python}", "train.py"),
        objective_metric="val_bpb",
        objective_direction=ObjectiveDirection.MINIMIZE,
        qualification_metric=None,
        qualification_minimum=None,
        public_feedback_metrics=("val_bpb", "training_seconds"),
        metric_patterns={},
        final_holdout_command=("{python}", "train.py"),
        preferred_backend=ExecutionBackend.HYBRID_MODAL,
    )
    spec = FactorialSpec(
        **{
            **openevolve_v2_protocol().__dict__,
            "protocol_version": "2.1",
            "study_id": "nanogpt-calibration-test",
        }
    )
    output_archive = io.BytesIO()
    with zipfile.ZipFile(output_archive, "w") as archive:
        archive.writestr("evaluation.stdout.log", "val_bpb: 1.234000\n")
        archive.writestr("evaluation.stderr.log", "")

    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeFunction:
        @staticmethod
        def from_name(
            app_name: str,
            function_name: str,
            *,
            environment_name: str | None,
        ) -> FakeFunction:
            calls.append(
                (
                    app_name,
                    function_name,
                    {"environment_name": environment_name},
                )
            )
            return FakeFunction()

        def remote(
            self,
            payload: bytes,
            task_payload: dict[str, object],
            timeout_seconds: int,
            run_seed: int | None,
            call_id: str,
        ) -> dict[str, object]:
            assert payload
            assert task_payload["adapter"] == NANOGPT_TASK_ADAPTER
            assert timeout_seconds == spec.budget.evaluator_timeout_seconds
            assert run_seed == spec.study_seed
            assert call_id
            return {
                "evaluation": {
                    "valid": True,
                    "fitness": -1.234,
                    "metrics": {
                        "val_bpb": 1.234,
                        "training_seconds": 300.0,
                    },
                    "evaluator_seconds": 300.0,
                    "evaluator_calls": 1,
                    "failure_kind": None,
                },
                "artifacts": output_archive.getvalue(),
                "worker_seconds": 301.0,
                "gpu_name": "NVIDIA H100 80GB HBM3",
            }

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=FakeFunction))
    calibration = prepare_calibration(
        tmp_path / "calibration",
        spec=spec,
        task=task_spec,
        repo_root=ROOT,
    )
    baseline_path = execute_calibration(
        calibration,
        spec=spec,
        task=task_spec,
        repo_root=ROOT,
        python_bin=sys.executable,
    )

    baseline = json.loads(baseline_path.read_text())
    usage = json.loads((calibration / "evaluation/modal-usage.json").read_text())
    assert calls == [
        (NANOGPT_APP_NAME, "evaluate_candidate", {"environment_name": None})
    ]
    assert baseline["calibration_kind"] == "executed_on_target_backend"
    assert baseline["metrics"]["val_bpb"] == pytest.approx(1.234)
    assert usage["record_kind"] == "baseline_calibration"
    assert usage["gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert (calibration / "modal-usage.jsonl").is_file()


def test_runtime_hash_change_fails_validation_and_execution(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    baseline = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=baseline,
        repo_root=ROOT,
        include_no_search=False,
    )
    assignment = json.loads((campaign / "schedule.json").read_text())[0]
    manifest_path = campaign / "runs" / assignment["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scientific_runtime_hash"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_campaign(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is False
    assert any(
        "scientific_runtime_hash mismatch" in error for error in report["errors"]
    )
    try:
        run_one_opportunity(
            campaign / "runs" / assignment["run_id"],
            spec=protocol(),
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary="not-used",
        )
    except ValueError as error:
        assert "scientific runtime changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("execution accepted a mismatched runtime")


def test_codex_event_usage_parser_uses_final_turn(tmp_path: Path) -> None:
    fake_codex = make_fake_codex(tmp_path / "fake-codex")
    workspace = make_seed(tmp_path / "workspace")
    result = CodexCli(str(fake_codex)).run(
        prompt="test",
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="one",
    )
    assert result.returncode == 0
    assert result.usage.input_tokens == 11
    assert result.usage.cached_input_tokens == 3
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 16


def test_controlled_openevolve_uses_vendor_prompt_and_diff_primitives(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("dacite") is None:
        import pytest

        pytest.skip("OpenEvolve dependencies live in architecture_discovery/.venv")
    workspace = make_seed(tmp_path / "workspace")
    visible = make_seed(tmp_path / "visible")
    adapter = OpenEvolveAdapter(
        CodexCli(str(make_fake_diff_codex(tmp_path / "fake-diff-codex"))),
        vendor_root=ROOT / "architecture_discovery/vendor/openevolve",
    )
    rendered = RenderedPrompt(
        text="controlled system prompt",
        common_template_sha256="a",
        search_state_sha256="b",
        proposal_policy_sha256="c",
        prompt_sha256="d",
        transition_active=False,
    )
    result = adapter.propose(
        rendered=rendered,
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="proposal-1",
        timeout_seconds=10,
        task=task(workspace),
        visible_workspaces=(visible,),
        selected_parent_id="seed",
        visible_records=({"candidate_id": "seed", "metrics": {"score": 0.0}},),
        run_seed=42,
    )
    assert result.adapter_error is None
    assert (workspace / "candidate.py").read_text() == "SCORE = 1\n"


def test_openevolve_v2_adapter_uses_bounded_neutral_strict_patch_prompt(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("dacite") is None:
        pytest.skip("OpenEvolve dependencies live in architecture_discovery/.venv")
    workspace = make_pair_token_seed(tmp_path / "workspace")
    visible = make_pair_token_seed(tmp_path / "visible")
    fake_codex = make_fake_v2_diff_codex(tmp_path / "fake-v2-diff-codex")
    adapter = OpenEvolveAdapter(
        CodexCli(str(fake_codex)),
        vendor_root=ROOT / "architecture_discovery/vendor/openevolve",
        v2=True,
        template_root=ROOT / "experiments/c0c3_factorial/templates/openevolve_v2",
    )
    rendered = RenderedPrompt(
        text="Optimize a learned transformer without external access.",
        common_template_sha256="a",
        search_state_sha256="b",
        proposal_policy_sha256="c",
        prompt_sha256="d",
        transition_active=False,
    )
    result = adapter.propose(
        rendered=rendered,
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="proposal-1",
        timeout_seconds=10,
        task=openevolve_v2_task(workspace),
        visible_workspaces=(visible,),
        selected_parent_id="seed",
        visible_records=(
            {
                "candidate_id": "seed",
                "metrics": {"accuracy": 1.0, "parameters": 1644},
            },
        ),
        run_seed=42,
        neutral_subject=True,
    )

    assert result.adapter_error is None
    assert result.mechanism == "change the training constant"
    assert result.evidence == "the supplied parent contains the old constant"
    assert (workspace / "src/train.py").read_text() == "VALUE = 2\n"
    prompt = fake_codex.with_suffix(".prompt.md").read_text()
    assert prompt.count("===== FILE: src/model.py =====") == 1
    assert "OpenEvolve" not in prompt
    assert "MECHANISM, HYPOTHESIS, INTENDED_EDIT, and EVIDENCE" in prompt


def test_openevolve_v21_adapter_uses_one_clean_prompt_contract(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("dacite") is None:
        pytest.skip("OpenEvolve dependencies live in architecture_discovery/.venv")
    workspace = make_pair_token_seed(tmp_path / "workspace")
    visible = make_pair_token_seed(tmp_path / "visible")
    reference = make_pair_token_seed(tmp_path / "reference")
    fake_codex = make_fake_v2_diff_codex(tmp_path / "fake-v21-diff-codex")
    adapter = OpenEvolveAdapter(
        CodexCli(str(fake_codex)),
        vendor_root=ROOT / "architecture_discovery/vendor/openevolve",
        v2=True,
        v21=True,
        template_root=ROOT / "experiments/c0c3_factorial/templates/openevolve_v2_1",
    )
    response_contract = "Return these short metadata lines"
    rendered = RenderedPrompt(
        text=(
            "Optimize a learned transformer without external access.\n"
            f"{response_contract}."
        ),
        common_template_sha256="a",
        search_state_sha256="b",
        proposal_policy_sha256="c",
        prompt_sha256="d",
        transition_active=False,
    )
    result = adapter.propose(
        rendered=rendered,
        workspace=workspace,
        model=ModelSpec("gpt-fake", "high"),
        log_root=tmp_path / "logs",
        call_id="proposal-1",
        timeout_seconds=10,
        task=openevolve_v21_task(workspace),
        visible_workspaces=(visible, reference),
        selected_parent_id="seed",
        visible_records=(
            {
                "candidate_id": "seed",
                "metrics": {"accuracy": 1.0, "parameters": 1644},
            },
            {
                "candidate_id": "reference",
                "metrics": {"accuracy": 0.995, "parameters": 1600},
            },
        ),
        run_seed=42,
        neutral_subject=True,
        artifact_clean_subject=True,
    )

    assert result.adapter_error is None
    prompt = fake_codex.with_suffix(".prompt.md").read_text()
    assert prompt.count(response_contract) == 1
    assert prompt.count("===== FILE: src/model.py =====") == 2
    assert "REFERENCE DESIGN 1" in prompt
    assert "VERIFIED VALUES" not in prompt
    assert ".design-references" not in prompt
    assert "followed by one or more exact SEARCH/REPLACE blocks" not in prompt


def test_v17_campaign_is_source_only_c0c3_and_launch_valid(
    tmp_path: Path,
) -> None:
    seed_source = make_pair_token_seed(tmp_path / "source")
    spec = FactorialSpec(
        protocol_version="1.7",
        study_id="artifact-clean-launch-test",
        study_seed=20260824,
        blocks=1,
        portfolio_capacity=4,
        transition_opportunities=(1,),
        conversation_mode=ConversationMode.CONTINUOUS,
        model=ModelSpec("gpt-fake", "xhigh"),
        budget=BudgetSpec(
            proposals=1,
            candidate_evaluations=1,
            max_total_tokens=10,
            max_evaluator_seconds=100.0,
            evaluator_timeout_seconds=10,
        ),
        execution_rule=STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
        include_no_search=False,
    )
    task_spec = artifact_clean_task(seed_source)
    framework_spec = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_confined_session_resume_v2",
        prompt_profile=AUTORESEARCH_V17_PROMPT_PROFILE,
        edit_mode="direct_workspace",
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task_spec,
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task_spec,
        framework=framework_spec,
        calibration_path=calibration,
        repo_root=ROOT,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task_spec,
        framework=framework_spec,
        repo_root=ROOT,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())

    assert report["valid"], report["errors"]
    assert len(schedule) == 4
    assert {row["condition"] for row in schedule} == {"C0", "C1", "C2", "C3"}
    for run in (campaign / "runs").iterdir():
        assert not (run / "task-support/checkpoints").exists()


def test_layer_b_is_sealed_until_completion_then_scores_factorial(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    try:
        export_layer_b_packets(campaign, spec=protocol(), task=task(seed_source))
    except RuntimeError as error:
        assert "sealed until every run completes" in str(error)
    else:  # pragma: no cover - fail explicitly if the seal regresses
        raise AssertionError("Layer B opened before search completed")
    schedule = json.loads((campaign / "schedule.json").read_text())
    for index, assignment in enumerate(schedule, start=1):
        run_dir = campaign / "runs" / assignment["run_id"]
        controller = SearchController.load(run_dir, protocol())
        controller.begin()
        seed_id = controller.state.incumbent_id
        shutil.copytree(
            run_dir / "candidates" / seed_id,
            run_dir / "candidates" / f"candidate-{index}",
        )
        controller.complete(
            candidate_id=f"candidate-{index}",
            artifact_path=f"candidates/{seed_id}",
            hypothesis=f"mechanism {index}",
            intended_edit="test one mechanism",
            evaluation=Evaluation(True, float(index), {"score": float(index)}, 0.1),
            usage=Usage(input_tokens=1, output_tokens=1),
            prompt_hashes={"prompt_sha256": f"hash-{index}"},
        )
    sealed = export_layer_b_packets(campaign, spec=protocol(), task=task(seed_source))
    packet_ids = [
        row["packet_id"]
        for row in csv.DictReader(
            (sealed / "packet_order.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    ]
    first_packet = sealed / "packets" / packet_ids[0]
    assert (first_packet / "parent/candidate.py").is_file()
    packet_payload = json.loads((first_packet / "packet.json").read_text())
    assert "layer_a_metrics" not in packet_payload
    assert packet_payload["layer_a_qualified"] is True
    annotations = sealed / "annotations.tsv"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "packet_id",
                "layer_b_qualified",
                "mechanism_cluster",
                "primary_mechanism",
                "reviewer",
                "notes",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for packet_id in packet_ids:
            writer.writerow(
                {
                    "packet_id": packet_id,
                    "layer_b_qualified": "1",
                    "mechanism_cluster": "cluster-a",
                    "primary_mechanism": "toy",
                    "reviewer": "test",
                    "notes": "",
                }
            )
    scored = score_layer_b(campaign, annotations_path=annotations)
    estimates = json.loads((scored / "factorial_estimates.json").read_text())
    assert estimates["cell_means"] == {"C0": 1.0, "C1": 1.0, "C2": 1.0, "C3": 1.0}
    assert estimates["portfolio_memory_main_effect"] == 0.0
    layer_c = run_layer_c(
        campaign,
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    summaries = json.loads((layer_c / "summary.json").read_text())
    assert len(summaries) == 5
    assert all(row["layer_c_evaluation"]["valid"] for row in summaries)
    no_search = next(row for row in summaries if row["condition"] == "N0")
    assert no_search["selected_by_layer_a_candidate_id"] == "candidate-5"
    assert no_search["selection_rule"] == (
        "post_search_best_independent_layer_a_candidate"
    )


def test_campaign_orchestration_is_blocked_round_robin(tmp_path: Path) -> None:
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    first = next_run(campaign, protocol())
    assert first is not None
    assert first.run_id == schedule[0]["run_id"]
    first_controller = SearchController.load(
        campaign / "runs" / first.run_id, protocol()
    )
    first_controller.begin()
    first_controller.complete(
        candidate_id="finished-first",
        artifact_path=f"candidates/{first_controller.state.incumbent_id}",
        hypothesis="test ordering",
        intended_edit="none",
        evaluation=Evaluation(False, None, {}, 0.0, evaluator_calls=0),
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_hashes={},
    )
    second = next_run(campaign, protocol())
    assert second is not None
    assert second.run_id == schedule[1]["run_id"]


def test_parallel_wave_selects_only_lagging_factorial_cells(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    schedule = json.loads((campaign / "schedule.json").read_text())
    first = next(row for row in schedule if row["condition"] != "N0")
    controller = SearchController.load(campaign / "runs" / first["run_id"], spec)
    controller.begin()
    controller.complete(
        candidate_id="interrupted-peer",
        artifact_path=f"candidates/{controller.state.incumbent_id}",
        hypothesis="simulate a peer completing before host interruption",
        intended_edit="none",
        evaluation=Evaluation(False, None, {}, 0.0, evaluator_calls=0),
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_hashes={},
    )
    wave = next_parallel_wave(campaign, spec)
    assert wave is not None
    assert wave.recovery_subset is True
    assert first["run_id"] not in {run.run_id for run in wave.factorial_runs}
    assert len(wave.factorial_runs) == 3
    assert wave.no_search_run is not None


def test_parallel_campaign_overlaps_c0_c3_and_serializes_n0(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_parallel_barrier_fake_codex(
        tmp_path / "fake-parallel-codex", tmp_path / "parallel-markers"
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is True
    assert report["controls"]["frozen_parallel_condition_rounds"] is True
    with pytest.raises(ValueError, match="serial campaign commands require"):
        next_run(campaign, spec)

    results = list(
        run_parallel_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
            max_block_rounds=1,
        )
    )
    assert len(results) == 1
    assert {row["condition"] for row in results[0]["factorial_records"]} == {
        "C0",
        "C1",
        "C2",
        "C3",
    }
    assert results[0]["no_search_record"]["condition"] == "N0"
    assert len(list((tmp_path / "parallel-markers").iterdir())) == 4
    schedule = json.loads((campaign / "schedule.json").read_text())
    assert all(
        SearchController.load(campaign / "runs" / row["run_id"], spec).state.status
        == "completed"
        for row in schedule
    )
    events = [
        json.loads(line)
        for line in (campaign / "parallel-rounds.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in events] == [
        "parallel_wave_started",
        "parallel_wave_completed",
    ]


def test_parallel_campaign_stops_after_zero_token_provider_failure(
    tmp_path: Path,
) -> None:
    spec = parallel_protocol()
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    failing_codex = make_failing_fake_codex(tmp_path / "failing-codex")

    with pytest.raises(ParallelWaveError, match="before token accounting"):
        list(
            run_parallel_campaign(
                campaign,
                spec=spec,
                task=task(seed_source),
                framework=framework(),
                repo_root=ROOT,
                python_bin=sys.executable,
                codex_binary=str(failing_codex),
                codex_timeout_seconds=10,
            )
        )

    schedule = json.loads((campaign / "schedule.json").read_text())
    states = {
        row["condition"]: SearchController.load(
            campaign / "runs" / row["run_id"], spec
        ).state
        for row in schedule
    }
    assert all(
        states[name].proposals_used == 1 for name in ("C0", "C1", "C2", "C3")
    )
    assert states["N0"].proposals_used == 0
    events = [
        json.loads(line)
        for line in (campaign / "parallel-rounds.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in events] == [
        "parallel_wave_started",
        "parallel_wave_failed",
    ]


def test_staged_campaign_runs_primary_only_and_preserves_later_stages(
    tmp_path: Path,
) -> None:
    spec = staged_protocol()
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_parallel_barrier_fake_codex(
        tmp_path / "fake-staged-codex", tmp_path / "staged-markers"
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task(seed_source),
        framework=framework(),
        repo_root=ROOT,
    )
    assert report["valid"] is True
    assert report["controls"]["frozen_staged_parallel_trajectories"] is True
    assert report["controls"]["frozen_blocked_round_robin_execution"] is False
    with pytest.raises(ValueError, match="parallel campaign commands require"):
        next_parallel_wave(campaign, spec)
    with pytest.raises(ValueError, match="complete the frozen primary stage"):
        next_staged_parallel_wave(
            campaign,
            spec,
            block=1,
            stage=NO_SEARCH_STAGE,
        )

    primary = list(
        run_staged_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            block=1,
            stage=FACTORIAL_STAGE,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
        )
    )
    assert len(primary) == 1
    assert primary[0]["execution_stage"] == "block-01-factorial"
    assert primary[0]["no_search_record"] is None
    assert {row["condition"] for row in primary[0]["factorial_records"]} == {
        "C0",
        "C1",
        "C2",
        "C3",
    }

    schedule = json.loads((campaign / "schedule.json").read_text())
    states = {
        (int(row["block"]), str(row["condition"])): SearchController.load(
            campaign / "runs" / row["run_id"], spec
        ).state
        for row in schedule
    }
    assert all(
        states[(1, condition)].status == "completed"
        for condition in ("C0", "C1", "C2", "C3")
    )
    assert states[(1, "N0")].proposals_used == 0
    assert all(
        states[(2, condition)].proposals_used == 0
        for condition in ("C0", "C1", "C2", "C3", "N0")
    )
    assert (
        next_staged_parallel_wave(
            campaign,
            spec,
            block=1,
            stage=FACTORIAL_STAGE,
        )
        is None
    )

    n0 = list(
        run_staged_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            block=1,
            stage=NO_SEARCH_STAGE,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
        )
    )
    assert len(n0) == 1
    assert n0[0]["factorial_records"] == []
    assert n0[0]["no_search_record"]["condition"] == "N0"

    extension = list(
        run_staged_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=framework(),
            repo_root=ROOT,
            python_bin=sys.executable,
            block=2,
            stage=FACTORIAL_STAGE,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
        )
    )
    assert len(extension) == 1
    assert extension[0]["execution_stage"] == "block-02-factorial"
    assert states[(2, "N0")].proposals_used == 0

    sealed = export_layer_b_packets(
        campaign,
        spec=spec,
        task=task(seed_source),
    )
    scope = json.loads((sealed / "scope.json").read_text())
    assert len(scope["run_ids"]) == 9
    assert any(run_id.endswith("-b01-n0") for run_id in scope["run_ids"])
    assert any("-b02-" in run_id for run_id in scope["run_ids"])
    assert not any(run_id.endswith("-b02-n0") for run_id in scope["run_ids"])
    with pytest.raises(ValueError, match="before primary Layer B/C"):
        next_staged_parallel_wave(
            campaign,
            spec,
            block=2,
            stage=NO_SEARCH_STAGE,
        )


def test_independent_staged_campaign_runs_trajectories_without_round_barriers(
    tmp_path: Path,
) -> None:
    spec = independent_staged_protocol()
    seed_source = make_seed(tmp_path / "source")
    fake_codex = make_independent_continuous_fake_codex(
        tmp_path / "fake-independent-continuous-codex",
        tmp_path / "independent-markers",
    )
    continuous_framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_session_resume_v1",
        prompt_profile="controlled_factorial_continuous_v1",
        edit_mode="direct_workspace",
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=continuous_framework,
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=task(seed_source),
        framework=continuous_framework,
        repo_root=ROOT,
    )
    assert report["valid"] is True
    assert report["controls"]["frozen_staged_independent_trajectories"] is True
    assert len(
        staged_independent_trajectories(
            campaign,
            spec,
            block=1,
            stage=FACTORIAL_STAGE,
        )
    ) == 4
    with pytest.raises(ValueError, match="staged parallel campaign commands require"):
        next_staged_parallel_wave(
            campaign,
            spec,
            block=1,
            stage=FACTORIAL_STAGE,
        )

    results = list(
        run_staged_independent_campaign(
            campaign,
            spec=spec,
            task=task(seed_source),
            framework=continuous_framework,
            repo_root=ROOT,
            python_bin=sys.executable,
            block=1,
            stage=FACTORIAL_STAGE,
            codex_binary=str(fake_codex),
            codex_timeout_seconds=10,
        )
    )
    assert len(results) == 4
    assert {result["condition"] for result in results} == {"C0", "C1", "C2", "C3"}
    assert all(result["status"] == "completed" for result in results)
    assert all(result["proposals_used"] == 2 for result in results)
    assert len(list((tmp_path / "independent-markers" / "initial").iterdir())) == 4
    assert (tmp_path / "independent-markers" / "resumed").is_dir()

    schedule = json.loads((campaign / "schedule.json").read_text())
    states = {
        (int(row["block"]), str(row["condition"])): SearchController.load(
            campaign / "runs" / row["run_id"], spec
        ).state
        for row in schedule
    }
    assert states[(1, "N0")].proposals_used == 0
    assert all(
        states[(2, condition)].proposals_used == 0
        for condition in ("C0", "C1", "C2", "C3", "N0")
    )
    events = [
        json.loads(line)
        for line in (campaign / "independent-trajectories.jsonl")
        .read_text()
        .splitlines()
    ]
    assert events[0]["event"] == "independent_trajectory_batch_started"
    assert events[-1]["event"] == "independent_trajectory_batch_completed"
    assert sum(
        event["event"] == "independent_trajectory_completed" for event in events
    ) == 4


def test_independent_staged_campaign_stops_after_zero_token_provider_failure(
    tmp_path: Path,
) -> None:
    spec = independent_staged_protocol()
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    continuous_framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_session_resume_v1",
        prompt_profile="controlled_factorial_continuous_v1",
        edit_mode="direct_workspace",
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=task(seed_source),
        framework=continuous_framework,
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    failing_codex = make_failing_fake_codex(tmp_path / "failing-codex")

    with pytest.raises(IndependentTrajectoryError):
        list(
            run_staged_independent_campaign(
                campaign,
                spec=spec,
                task=task(seed_source),
                framework=continuous_framework,
                repo_root=ROOT,
                python_bin=sys.executable,
                block=1,
                stage=FACTORIAL_STAGE,
                codex_binary=str(failing_codex),
                codex_timeout_seconds=10,
            )
        )

    schedule = json.loads((campaign / "schedule.json").read_text())
    states = {
        (int(row["block"]), str(row["condition"])): SearchController.load(
            campaign / "runs" / row["run_id"], spec
        ).state
        for row in schedule
    }
    assert all(
        states[(1, condition)].proposals_used <= 1
        and states[(1, condition)].active is None
        for condition in ("C0", "C1", "C2", "C3")
    )
    assert states[(1, "N0")].proposals_used == 0
    events = [
        json.loads(line)
        for line in (campaign / "independent-trajectories.jsonl")
        .read_text()
        .splitlines()
    ]
    assert events[0]["event"] == "independent_trajectory_batch_started"
    assert events[-1]["event"] == "independent_trajectory_batch_failed"


def test_individually_controlled_trajectory_can_pause_and_resume_without_peers(
    tmp_path: Path,
) -> None:
    spec = individually_controlled_staged_protocol()
    seed_source = make_neutral_seed(tmp_path / "source")
    neutral_framework = FrameworkSpec(
        framework_id=FrameworkKind.AUTORESEARCH,
        adapter="codex_direct_editor_session_resume_v1",
        prompt_profile=NEUTRAL_PROMPT_PROFILE,
        edit_mode="direct_workspace",
    )
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=spec,
        task=neutral_task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=spec,
        task=neutral_task(seed_source),
        framework=neutral_framework,
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=True,
    )
    report = validate_campaign(
        campaign,
        spec=spec,
        task=neutral_task(seed_source),
        framework=neutral_framework,
        repo_root=ROOT,
    )
    assert report["valid"] is True
    assert report["controls"]["frozen_staged_individually_controlled_trajectories"]

    schedule = json.loads((campaign / "schedule.json").read_text())
    c0 = next(row for row in schedule if row["condition"] == "C0")
    c1 = next(row for row in schedule if row["condition"] == "C1")
    n0 = next(row for row in schedule if row["condition"] == "N0")
    c0_codex = make_slow_neutral_continuous_fake_codex(tmp_path / "c0-codex")
    c0_result: dict[str, object] = {}

    def start_c0() -> None:
        c0_result.update(
            run_staged_individual_trajectory(
                campaign,
                spec=spec,
                task=neutral_task(seed_source),
                framework=neutral_framework,
                repo_root=ROOT,
                python_bin=sys.executable,
                run_id=str(c0["run_id"]),
                codex_binary=str(c0_codex),
                codex_timeout_seconds=10,
            )
        )

    worker = threading.Thread(target=start_c0)
    worker.start()
    deadline = time.monotonic() + 5
    while (
        SearchController.load(campaign / "runs" / c0["run_id"], spec).state.active
        is None
    ):
        if time.monotonic() >= deadline:
            raise AssertionError("C0 never began its first opportunity")
        time.sleep(0.01)
    pause = request_staged_trajectory_pause(
        campaign,
        spec=spec,
        run_id=str(c0["run_id"]),
        reason="test cooperative pause",
    )
    assert pause["status"] == "pause_requested"
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert c0_result["status"] == "paused"
    assert c0_result["proposals_used"] == 1

    resumed = run_staged_individual_trajectory(
        campaign,
        spec=spec,
        task=neutral_task(seed_source),
        framework=neutral_framework,
        repo_root=ROOT,
        python_bin=sys.executable,
        run_id=str(c0["run_id"]),
        resume=True,
        codex_binary=str(c0_codex),
        codex_timeout_seconds=10,
    )
    assert resumed["status"] == "completed"
    assert resumed["proposals_used"] == 2

    # The old campaign writer lock does not serialize the v1.5 peer command.
    c1_codex = make_slow_neutral_continuous_fake_codex(tmp_path / "c1-codex")
    with campaign_lock(campaign):
        c1_result = run_staged_individual_trajectory(
            campaign,
            spec=spec,
            task=neutral_task(seed_source),
            framework=neutral_framework,
            repo_root=ROOT,
            python_bin=sys.executable,
            run_id=str(c1["run_id"]),
            codex_binary=str(c1_codex),
            codex_timeout_seconds=10,
        )
    assert c1_result["status"] == "completed"

    with pytest.raises(ValueError, match="complete the frozen primary stage"):
        run_staged_individual_trajectory(
            campaign,
            spec=spec,
            task=neutral_task(seed_source),
            framework=neutral_framework,
            repo_root=ROOT,
            python_bin=sys.executable,
            run_id=str(n0["run_id"]),
            codex_binary=str(c1_codex),
            codex_timeout_seconds=10,
        )

    lifecycle = [
        json.loads(line)
        for line in (campaign / "trajectory-lifecycle.jsonl").read_text().splitlines()
    ]
    assert "trajectory_pause_requested" in {event["event"] for event in lifecycle}
    assert "trajectory_paused" in {event["event"] for event in lifecycle}
    assert "trajectory_resumed" in {event["event"] for event in lifecycle}
    assert {event["run_id"] for event in lifecycle} >= {
        str(c0["run_id"]),
        str(c1["run_id"]),
    }


def test_interrupted_opportunity_recovery_consumes_proposal_not_evaluation(
    tmp_path: Path,
) -> None:
    seed_source = make_seed(tmp_path / "source")
    calibration = calibrate_task(
        tmp_path / "calibration",
        spec=protocol(),
        task=task(seed_source),
        repo_root=ROOT,
        python_bin=sys.executable,
    )
    campaign = create_campaign(
        tmp_path / "campaign",
        spec=protocol(),
        task=task(seed_source),
        framework=framework(),
        calibration_path=calibration,
        repo_root=ROOT,
        include_no_search=False,
    )
    assignment = json.loads((campaign / "schedule.json").read_text())[0]
    run_dir = campaign / "runs" / assignment["run_id"]
    SearchController.load(run_dir, protocol()).begin()
    record = recover_active_opportunity(
        run_dir, spec=protocol(), reason="simulated host interruption"
    )
    assert record["evaluation"]["failure_kind"] == "infrastructure_interruption"
    assert record["evaluator_calls_increment"] == 0
    controller = SearchController.load(run_dir, protocol())
    assert controller.state.proposals_used == 1
    assert controller.state.evaluations_used == 0
    assert controller.state.status == "completed"
