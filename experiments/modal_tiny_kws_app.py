"""Modal CPU worker: 2 cores and 4 GiB PER evaluator, no application slot cap."""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import modal

from experiments.c0c3_factorial.tiny_v21_runtime import archive_outputs, extract_inputs

APP_NAME = "rl4rl-tiny-kws-cpu"

LOCAL_REPO = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(
        str(LOCAL_REPO / "experiments/c0c3_factorial"),
        "/opt/rl4rl/experiments/c0c3_factorial",
        copy=True,
        ignore=["**/__pycache__", "**/.pytest_cache"],
    )
    .add_local_dir(
        str(LOCAL_REPO / "data/raw/tiny-kws-rnn"),
        "/opt/rl4rl/data/raw/tiny-kws-rnn",
        copy=True,
        ignore=["extracted/**", "*.partial-*"],
    )
    .add_local_file(
        str(Path(__file__).resolve()),
        "/opt/rl4rl/experiments/modal_tiny_kws_app.py",
        copy=True,
    )
    .env({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    .workdir("/opt/rl4rl")
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=35 * 60,
    scaledown_window=60,
    retries=0,
    include_source=False,
)
def evaluate_candidate(payload, task_payload, timeout_seconds, run_seed, call_id):
    from experiments.c0c3_factorial.evaluator import CommandEvaluator
    from experiments.c0c3_factorial.spec import (
        ExecutionBackend,
        ObjectiveDirection,
        TaskSpec,
    )

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="tiny-kws-") as temporary:
        root = Path(temporary)
        extract_inputs(payload, root)
        converted = dict(task_payload)
        for key in (
            "editable_paths",
            "evaluator_command",
            "public_feedback_metrics",
            "final_holdout_command",
        ):
            converted[key] = tuple(converted[key])
        converted["objective_direction"] = ObjectiveDirection(
            converted["objective_direction"]
        )
        converted["preferred_backend"] = ExecutionBackend.LOCAL
        task = TaskSpec(**converted)
        evaluator = CommandEvaluator(
            task=task,
            support_source=root / "support",
            repo_root=Path("/opt/rl4rl"),
            python_bin="python",
        )
        output = root / "opportunity"
        result = evaluator.evaluate(
            candidate_snapshot=root / "candidate",
            opportunity_root=output,
            timeout_seconds=timeout_seconds,
            run_seed=run_seed,
        )
        return dict(
            call_id=call_id,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            evaluation=asdict(result.evaluation),
            artifacts=archive_outputs(output),
            worker_seconds=time.monotonic() - started,
            cpu=2,
            memory_mib=4096,
        )
