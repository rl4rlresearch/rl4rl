"""Task-scoped Modal-first transport with a hard three-process CPU fallback."""

from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from .evaluator import (
    CommandEvaluator,
    EvaluationArtifacts,
    _release_slot,
    _try_acquire_slot,
    windows_evaluator_root,
)
from .hybrid_evaluator import (
    MAX_ARCHIVE_BYTES,
    ModalCommandEvaluator,
    _archive_inputs,
    _extract_outputs,
    _stable_remote_call_id,
)
from .state import Evaluation, append_jsonl, atomic_json

APP_NAME = "rl4rl-tiny-adderboard-v21"


def prepare_seed_workspace(*, source, destination, task, **_kwargs):
    destination.mkdir(parents=True, exist_ok=False)
    for relative in task.editable_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def extract_inputs(payload: bytes, destination: Path) -> None:
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
                raise ValueError("unsafe evaluator archive")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))


def archive_outputs(root: Path) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in (
            "evaluation.json",
            "evaluation.stdout.log",
            "evaluation.stderr.log",
            "evaluation-workspace/checkpoints/best.pt",
            "evaluation-workspace/checkpoints/last.pt",
        ):
            path = root / relative
            if path.is_file() and not path.is_symlink():
                archive.writestr(relative, path.read_bytes())
    result = stream.getvalue()
    if len(result) > MAX_ARCHIVE_BYTES:
        raise ValueError("output archive exceeds 16 MiB")
    return result


class ModalFallbackEvaluator(CommandEvaluator):
    def __init__(self, *, options, **kwargs):
        # The old per-block/global schedulers must never limit Modal dispatch.
        clean = {
            key: kwargs[key]
            for key in ("task", "support_source", "repo_root", "python_bin")
        }
        super().__init__(**clean)
        self.options = options
        self.fallback_root = windows_evaluator_root()
        self.local_capacity = int(options.get("local_evaluator_capacity", 3))
        if self.local_capacity != 3:
            raise ValueError("this campaign freezes Windows fallback capacity at three")

    @contextlib.contextmanager
    def _evaluation_slot(self, opportunity_root, **kwargs):
        # evaluate() already holds the shared Windows lease before local work.
        # Modal dispatch never enters a local scheduler.
        yield

    def _remote(
        self, *, candidate_snapshot, opportunity_root, timeout_seconds, run_seed
    ):
        import modal

        payload = _archive_inputs(self.support_source, candidate_snapshot)
        call_id = _stable_remote_call_id(
            task_id=self.task.task_id,
            candidate_snapshot=candidate_snapshot,
            opportunity_root=opportunity_root,
        )
        started = time.monotonic()
        call = None
        receipt = {
            "call_id": call_id,
            "backend": "modal",
            "cpu": 2,
            "memory_mib": 4096,
            "gpu": None,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            task_payload = asdict(self.task)
            if self.task.adapter == "uci_har_source_only_v1":
                expected = hashlib.sha256(
                    (
                        self.repo_root / "experiments/c0c3_factorial/uci_har.py"
                    ).read_bytes()
                ).hexdigest()
                task_payload["extension_options"]["expected_evaluator_sha256"] = (
                    expected
                )
                receipt["expected_evaluator_sha256"] = expected
            function = modal.Function.from_name(
                self.options.get("modal_app", APP_NAME), "evaluate_candidate"
            )
            call = function.spawn(
                payload, task_payload, timeout_seconds, run_seed, call_id
            )
            receipt["modal_function_call_id"] = call.object_id
            receipt["status"] = "dispatched"
            atomic_json(opportunity_root / "remote-dispatch.json", receipt)
            response = call.get(timeout=timeout_seconds + 120)
            if (
                response.get("call_id") != call_id
                or response.get("payload_sha256") != receipt["payload_sha256"]
            ):
                raise RuntimeError("remote evaluation identity mismatch")
            evaluation = Evaluation(**response["evaluation"])
            _extract_outputs(response["artifacts"], opportunity_root)
            receipt.update(
                status="completed", worker_seconds=response["worker_seconds"]
            )
            atomic_json(opportunity_root / "remote-dispatch.json", receipt)
            ModalCommandEvaluator._record_usage(
                opportunity_root,
                call_id=call_id,
                local_wall_seconds=time.monotonic() - started,
                worker_seconds=response["worker_seconds"],
                gpu_name="CPU (2 cores, 4 GiB)",
                status="completed",
                app_name=self.options.get("modal_app", APP_NAME),
                function_name="evaluate_candidate",
            )
            return EvaluationArtifacts(
                evaluation,
                opportunity_root / "evaluation.stdout.log",
                opportunity_root / "evaluation.stderr.log",
                opportunity_root / "evaluation-workspace",
            )
        except Exception as error:
            receipt.update(
                status="infrastructure_failure",
                error=f"{type(error).__name__}: {error}",
            )
            if call is not None:
                # Finish/cancel the known remote call before any local attempt.
                try:
                    call.cancel(terminate_containers=True)
                    receipt["cancel_requested"] = True
                except Exception as cancel_error:
                    receipt["cancel_error"] = str(cancel_error)
                    atomic_json(opportunity_root / "remote-dispatch.json", receipt)
                    # An unconfirmed remote execution must not silently duplicate.
                    raise RuntimeError(
                        "could not confirm cancellation of remote evaluation"
                    ) from error
            atomic_json(opportunity_root / "remote-dispatch.json", receipt)
            append_jsonl(opportunity_root / "backend-attempts.jsonl", receipt)
            return None

    def evaluate(
        self,
        *,
        candidate_snapshot,
        opportunity_root,
        timeout_seconds,
        run_seed=None,
        verify_existing_checkpoint=False,
    ):
        if verify_existing_checkpoint:
            raise ValueError("v2.1 calibration trains the seed from scratch")
        opportunity_root.mkdir(parents=True, exist_ok=True)
        arguments = dict(
            candidate_snapshot=candidate_snapshot,
            opportunity_root=opportunity_root,
            timeout_seconds=timeout_seconds,
            run_seed=run_seed,
        )
        next_remote = 0.0
        queued = time.monotonic()
        self.fallback_root.mkdir(parents=True, exist_ok=True)
        while True:
            # Each new proposal tries Modal; queued fallback work retries too.
            if time.monotonic() >= next_remote:
                remote = self._remote(**arguments)
                if remote is not None:
                    # Candidate failures are results, never backend-fallback triggers.
                    return remote
                next_remote = time.monotonic() + float(
                    self.options.get("modal_retry_seconds", 30)
                )
            lease = _try_acquire_slot(
                root=self.fallback_root,
                capacity=3,
                first=0,
                opportunity_root=opportunity_root,
                scope="windows_fallback",
            )
            if lease is not None:
                try:
                    atomic_json(
                        opportunity_root / "evaluator-queue.json",
                        {
                            "backend": "windows_fallback",
                            "capacity": 3,
                            "slot": str(lease.path),
                            "wait_seconds": time.monotonic() - queued,
                        },
                    )
                    result = super().evaluate(**arguments)
                    atomic_json(
                        opportunity_root / "backend.json",
                        {
                            "backend": "windows_fallback",
                            "cpu_threads": 2,
                            "reason": "Modal infrastructure unavailable",
                        },
                    )
                    return result
                finally:
                    _release_slot(lease)
            atomic_json(
                opportunity_root / "evaluator-queue.json",
                {
                    "backend": "waiting_for_windows_or_modal",
                    "capacity": 3,
                    "wait_seconds": time.monotonic() - queued,
                },
            )
            time.sleep(0.5)


def make_evaluator(*, options, **kwargs):
    return ModalFallbackEvaluator(options=options, **kwargs)
