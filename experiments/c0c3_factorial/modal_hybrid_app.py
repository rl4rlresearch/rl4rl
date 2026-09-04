"""Deployed Modal L4 worker for evaluator-only protocol-2.0/2.1 execution."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

try:
    import modal
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    modal = None

from .hybrid_evaluator import APP_NAME, MAX_ARCHIVE_BYTES

REMOTE_REPO = Path("/opt/rl4rl")
LOCAL_REPO = Path(__file__).resolve().parents[2]


def _extract_inputs(payload: bytes, destination: Path) -> None:
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("input archive exceeds 16 MiB")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in {"support", "candidate"}
            ):
                raise ValueError("unsafe evaluator input archive")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


def _archive_outputs(root: Path) -> bytes:
    selected = (
        "evaluation.json",
        "evaluation.stdout.log",
        "evaluation.stderr.log",
        "evaluation-workspace/checkpoints/best.pt",
        "evaluation-workspace/checkpoints/last.pt",
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in selected:
            path = root / relative
            if path.is_file() and not path.is_symlink():
                archive.writestr(relative, path.read_bytes())
        for pattern in (
            "evaluation-workspace/results/runs/*/metrics.csv",
            "evaluation-workspace/results/runs/*/summary.json",
            "evaluation-workspace/results/runs/*/config.json",
        ):
            for path in sorted(root.glob(pattern)):
                if path.is_file() and not path.is_symlink():
                    archive.writestr(
                        path.relative_to(root).as_posix(), path.read_bytes()
                    )
    payload = stream.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("output archive exceeds 16 MiB")
    return payload


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128"
        )
        .add_local_dir(
            str(LOCAL_REPO / "experiments"),
            remote_path=str(REMOTE_REPO / "experiments"),
            copy=True,
            ignore=["**/__pycache__", "**/.pytest_cache", "**/data"],
        )
        .add_local_dir(
            str(LOCAL_REPO / "architecture_discovery/vendor/AdderBoard"),
            remote_path=str(REMOTE_REPO / "architecture_discovery/vendor/AdderBoard"),
            copy=True,
            ignore=["**/__pycache__"],
        )
        .workdir(str(REMOTE_REPO))
    )
    app = modal.App(APP_NAME)
    result_cache = modal.Dict.from_name(
        "rl4rl-c0c3-evaluator-results-v3", create_if_missing=True
    )

    @app.function(
        image=image,
        gpu="L4",
        cpu=4,
        memory=16384,
        timeout=35 * 60,
        max_containers=3,
        scaledown_window=300,
        retries=0,
        include_source=False,
    )
    def evaluate_candidate(
        payload: bytes,
        task_payload: dict[str, object],
        timeout_seconds: int,
        run_seed: int | None,
        call_id: str,
    ) -> dict[str, object]:
        import torch

        from experiments.c0c3_factorial.evaluator import CommandEvaluator
        from experiments.c0c3_factorial.spec import (
            ExecutionBackend,
            ObjectiveDirection,
            TaskSpec,
        )

        payload_sha256 = hashlib.sha256(payload).hexdigest()
        cached = result_cache.get(call_id)
        if cached is not None:
            if cached.get("payload_sha256") != payload_sha256:
                raise RuntimeError("stable evaluator call ID was reused with new input")
            if cached.get("status") == "completed":
                return cached["response"]
            deadline = time.monotonic() + timeout_seconds
            while cached.get("status") == "running" and time.monotonic() < deadline:
                time.sleep(1)
                cached = result_cache.get(call_id)
            if cached.get("status") == "completed":
                return cached["response"]
            raise RuntimeError(f"existing evaluator call is {cached.get('status')}")
        result_cache[call_id] = {
            "status": "running",
            "payload_sha256": payload_sha256,
        }
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"c0c3-{call_id}-") as temporary:
            root = Path(temporary)
            _extract_inputs(payload, root)
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
            output = root / "opportunity"
            evaluator = CommandEvaluator(
                task=task,
                support_source=root / "support",
                repo_root=REMOTE_REPO,
                python_bin="python",
            )
            artifacts = evaluator.evaluate(
                candidate_snapshot=root / "candidate",
                opportunity_root=output,
                timeout_seconds=timeout_seconds,
                run_seed=run_seed,
            )
            response = {
                "schema_version": "1.0",
                "call_id": call_id,
                "payload_sha256": payload_sha256,
                "evaluation": asdict(artifacts.evaluation),
                "artifacts": _archive_outputs(output),
                "worker_seconds": time.monotonic() - started,
                "gpu_name": torch.cuda.get_device_name(0),
            }
            result_cache[call_id] = {
                "status": "completed",
                "payload_sha256": payload_sha256,
                "response": response,
            }
            return response
else:
    image = None
    app = None
    evaluate_candidate = None
    result_cache = None


def main() -> None:
    print(json.dumps({"app": APP_NAME, "function": "evaluate_candidate"}))


if __name__ == "__main__":
    main()
