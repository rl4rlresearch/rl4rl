"""Local Codex / deployed-Modal evaluator transport for protocol 2.0."""

from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

from .evaluator import CommandEvaluator, EvaluationArtifacts
from .state import Evaluation, append_jsonl

APP_NAME = "rl4rl-c0c3-hybrid-evaluator-v2"
FUNCTION_NAME = "evaluate_candidate"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


def _archive_inputs(
    support_source: Path,
    candidate_snapshot: Path,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, root in (
            ("support", support_source),
            ("candidate", candidate_snapshot),
        ):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                archive.writestr(f"{label}/{relative}", path.read_bytes())
    payload = stream.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("hybrid evaluator input archive exceeds 16 MiB")
    return payload


def _extract_outputs(payload: bytes, destination: Path) -> None:
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("hybrid evaluator output archive exceeds 16 MiB")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("hybrid evaluator returned an unsafe archive path")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


class ModalCommandEvaluator(CommandEvaluator):
    """Send only training/verification to a deployed Modal L4 function."""

    def evaluate(
        self,
        *,
        candidate_snapshot: Path,
        opportunity_root: Path,
        timeout_seconds: int,
        run_seed: int | None = None,
        verify_existing_checkpoint: bool = False,
    ) -> EvaluationArtifacts:
        if verify_existing_checkpoint:
            raise ValueError(
                "hybrid calibration must execute the frozen training path; "
                "checkpoint-only verification is not supported"
            )
        try:
            import modal
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Modal SDK is required for preferred_backend=hybrid_modal"
            ) from error

        payload = _archive_inputs(self.support_source, candidate_snapshot)
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        with self._evaluation_slot(opportunity_root):
            try:
                function = modal.Function.from_name(
                    os.environ.get("C0C3_MODAL_APP", APP_NAME),
                    os.environ.get("C0C3_MODAL_FUNCTION", FUNCTION_NAME),
                    environment_name=os.environ.get("MODAL_ENVIRONMENT") or None,
                )
                response = function.remote(
                    payload,
                    asdict(self.task),
                    timeout_seconds,
                    run_seed,
                    call_id,
                )
            except Exception as error:  # noqa: BLE001 - record remote transport failure
                elapsed = time.monotonic() - started
                opportunity_root.mkdir(parents=True, exist_ok=True)
                stderr = opportunity_root / "evaluation.stderr.log"
                stdout = opportunity_root / "evaluation.stdout.log"
                workspace = opportunity_root / "evaluation-workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                stdout.write_text("", encoding="utf-8")
                stderr.write_text(
                    "Modal evaluator transport failed: "
                    f"{type(error).__name__}: {error}\n",
                    encoding="utf-8",
                )
                evaluation = Evaluation(
                    valid=False,
                    fitness=None,
                    metrics={
                        "remote_call_id": call_id,
                        "remote_error": f"{type(error).__name__}: {error}",
                    },
                    evaluator_seconds=elapsed,
                    evaluator_calls=1,
                    failure_kind="remote_infrastructure",
                )
                self._record_usage(
                    opportunity_root,
                    call_id=call_id,
                    local_wall_seconds=elapsed,
                    worker_seconds=None,
                    gpu_name=None,
                    status="failed",
                )
                return EvaluationArtifacts(evaluation, stdout, stderr, workspace)

        if not isinstance(response, dict):
            raise RuntimeError("hybrid evaluator returned a non-object response")
        artifact_archive = response.get("artifacts")
        evaluation_payload = response.get("evaluation")
        if not isinstance(artifact_archive, bytes) or not isinstance(
            evaluation_payload, dict
        ):
            raise RuntimeError("hybrid evaluator response is incomplete")
        _extract_outputs(artifact_archive, opportunity_root)
        evaluation = Evaluation(**evaluation_payload)
        elapsed = time.monotonic() - started
        self._record_usage(
            opportunity_root,
            call_id=call_id,
            local_wall_seconds=elapsed,
            worker_seconds=float(response.get("worker_seconds", 0.0)),
            gpu_name=str(response.get("gpu_name", "unknown")),
            status="completed",
        )
        return EvaluationArtifacts(
            evaluation=evaluation,
            stdout_path=opportunity_root / "evaluation.stdout.log",
            stderr_path=opportunity_root / "evaluation.stderr.log",
            workspace_path=opportunity_root / "evaluation-workspace",
        )

    @staticmethod
    def _record_usage(
        opportunity_root: Path,
        *,
        call_id: str,
        local_wall_seconds: float,
        worker_seconds: float | None,
        gpu_name: str | None,
        status: str,
    ) -> None:
        campaign = opportunity_root.parents[3]
        record = {
            "schema_version": "1.0",
            "call_id": call_id,
            "run_id": opportunity_root.parents[1].name,
            "opportunity": int(opportunity_root.name),
            "app": os.environ.get("C0C3_MODAL_APP", APP_NAME),
            "function": os.environ.get("C0C3_MODAL_FUNCTION", FUNCTION_NAME),
            "status": status,
            "local_wall_seconds": local_wall_seconds,
            "worker_seconds": worker_seconds,
            "gpu_name": gpu_name,
        }
        append_jsonl(campaign / "modal-usage.jsonl", record)
        (opportunity_root / "modal-usage.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
