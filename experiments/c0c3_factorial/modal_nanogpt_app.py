"""Dedicated H100 evaluator-only service for the pinned nanoGPT task."""

from __future__ import annotations

import io
import json
import subprocess
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

from .hybrid_evaluator import (
    MAX_ARCHIVE_BYTES,
    NANOGPT_APP_NAME,
)

REMOTE_REPO = Path("/opt/rl4rl")
REMOTE_AUTORESEARCH = Path("/opt/autoresearch")
LOCAL_REPO = Path(__file__).resolve().parents[2]
CACHE_VOLUME_NAME = "rl4rl-autoresearch-cache"


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
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in (
            "evaluation.json",
            "evaluation.stdout.log",
            "evaluation.stderr.log",
        ):
            path = root / relative
            if path.is_file() and not path.is_symlink():
                archive.writestr(relative, path.read_bytes())
    payload = stream.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("output archive exceeds 16 MiB")
    return payload


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "kernels>=0.11.7",
            "matplotlib>=3.10.8",
            "numpy>=2.2.6",
            "pandas>=2.3.3",
            "pyarrow>=21.0.0",
            "requests>=2.32.0",
            "rustbpe>=0.1.0",
            "tiktoken>=0.11.0",
            "torch==2.9.1",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .add_local_dir(
            str(LOCAL_REPO / "experiments"),
            remote_path=str(REMOTE_REPO / "experiments"),
            copy=True,
            ignore=["**/__pycache__", "**/.pytest_cache", "**/data"],
        )
        .add_local_dir(
            str(LOCAL_REPO / "architecture_discovery/vendor/autoresearch"),
            remote_path=str(REMOTE_AUTORESEARCH),
            copy=True,
            ignore=[".git", "**/__pycache__", "results", "queue"],
        )
        .workdir(str(REMOTE_REPO))
    )
    app = modal.App(NANOGPT_APP_NAME)
    cache = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        cpu=8,
        memory=32768,
        timeout=2 * 60 * 60,
        max_containers=1,
        retries=0,
        include_source=False,
        volumes={"/root/.cache/autoresearch": cache},
    )
    def prepare_cache(num_shards: int = 10) -> str:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        cache.reload()
        completed = subprocess.run(
            ["python", "prepare.py", "--num-shards", str(num_shards)],
            cwd=REMOTE_AUTORESEARCH,
            text=True,
            capture_output=True,
            check=False,
        )
        cache.commit()
        if completed.returncode:
            raise RuntimeError(
                f"nanoGPT cache preparation failed: {completed.stderr[-4000:]}"
            )
        return completed.stdout

    @app.function(
        image=image,
        gpu="H100",
        cpu=8,
        memory=65536,
        timeout=20 * 60,
        max_containers=3,
        scaledown_window=300,
        retries=0,
        include_source=False,
        volumes={
            "/root/.cache/autoresearch": cache.with_mount_options(read_only=True)
        },
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

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"nanogpt-{call_id}-") as temporary:
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
            return {
                "schema_version": "1.0",
                "call_id": call_id,
                "evaluation": asdict(artifacts.evaluation),
                "artifacts": _archive_outputs(output),
                "worker_seconds": time.monotonic() - started,
                "gpu_name": torch.cuda.get_device_name(0),
            }
else:
    image = None
    app = None
    cache = None
    prepare_cache = None
    evaluate_candidate = None


def main() -> None:
    print(json.dumps({"app": NANOGPT_APP_NAME, "function": "evaluate_candidate"}))


if __name__ == "__main__":
    main()
