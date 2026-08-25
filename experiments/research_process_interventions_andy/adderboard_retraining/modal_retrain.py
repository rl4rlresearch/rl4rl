"""Modal retraining for final incumbents from the 24 trajectory sweep.

The application bundles a read-only snapshot of the repository training
runtime and the versioned candidate JSON files into an isolated Modal image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNTIME_ROOT = (
    Path(
        os.environ.get(
            "RL4RL_RETRAIN_RUNTIME",
            str(REPO_ROOT / "architecture_discovery"),
        )
    )
    .expanduser()
    .resolve()
)
CANDIDATE_ROOT = HERE / "candidates"
REMOTE_RUNTIME = Path("/opt/rl4rl-runtime")
REMOTE_CANDIDATES = Path("/opt/rl4rl-candidates")
REMOTE_ADDERBOARD = Path("/opt/AdderBoard")
RESULT_ROOT = Path("/results")
APP_NAME = "rl4rl-small-architecture-retraining"
VOLUME_NAME = "rl4rl-small-architecture-retraining-results"
GPU = "T4"
MAX_PARALLEL_JOBS = 8
FUNCTION_TIMEOUT_SECONDS = 2_400
_SAFE_ID = re.compile(r"\A[a-z0-9][a-z0-9-]{0,79}\Z")


STAGES: dict[str, dict[str, int | float | bool]] = {
    "screen": {
        "steps": 1_000,
        "global_batch_size": 512,
        "warmup_steps": 100,
        "validation_interval": 100,
        "validation_examples": 512,
        "checkpoint_interval": 100,
        "maximum_wall_seconds": 900,
        "scientific": False,
    },
    "develop": {
        "steps": 5_000,
        "global_batch_size": 512,
        "warmup_steps": 300,
        "validation_interval": 500,
        "validation_examples": 1_000,
        "checkpoint_interval": 500,
        "maximum_wall_seconds": 1_800,
        "scientific": False,
    },
    "final": {
        # Retain the historical stage name as a compatibility alias, but keep
        # every production retraining launch at the requested 5k budget.
        "steps": 5_000,
        "global_batch_size": 512,
        "warmup_steps": 300,
        "validation_interval": 500,
        "validation_examples": 1_000,
        "checkpoint_interval": 500,
        "maximum_wall_seconds": 1_800,
        # This standalone lane is exploratory. Formal scientific qualification
        # should use the repository's frozen and approval-bound full profile.
        "scientific": False,
    },
}


def _require_local_inputs() -> None:
    required = (
        RUNTIME_ROOT / "common",
        RUNTIME_ROOT / "architecture_ir",
        RUNTIME_ROOT / "containment",
        RUNTIME_ROOT / "evaluation",
        RUNTIME_ROOT / "scripts",
        RUNTIME_ROOT / "vendor/AdderBoard/verify.py",
        RUNTIME_ROOT / "uv.lock",
        RUNTIME_ROOT / "experiment_manifest.yaml",
        CANDIDATE_ROOT / "manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = "standalone retraining inputs are missing: " + ", ".join(missing)
        raise RuntimeError(message)


if modal.is_local():
    _require_local_inputs()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_pip_install(
            "numpy==2.3.2",
            "pyyaml==6.0.2",
            "torch==2.7.1",
        )
        .add_local_dir(
            RUNTIME_ROOT / "common", str(REMOTE_RUNTIME / "common"), copy=True
        )
        .add_local_dir(
            RUNTIME_ROOT / "architecture_ir",
            str(REMOTE_RUNTIME / "architecture_ir"),
            copy=True,
        )
        .add_local_dir(
            RUNTIME_ROOT / "containment",
            str(REMOTE_RUNTIME / "containment"),
            copy=True,
        )
        .add_local_dir(
            RUNTIME_ROOT / "evaluation",
            str(REMOTE_RUNTIME / "evaluation"),
            copy=True,
        )
        .add_local_dir(
            RUNTIME_ROOT / "scripts", str(REMOTE_RUNTIME / "scripts"), copy=True
        )
        .add_local_dir(
            RUNTIME_ROOT / "vendor/AdderBoard",
            str(REMOTE_ADDERBOARD),
            copy=True,
        )
        .add_local_file(
            RUNTIME_ROOT / "uv.lock", str(REMOTE_RUNTIME / "uv.lock"), copy=True
        )
        .add_local_file(
            RUNTIME_ROOT / "experiment_manifest.yaml",
            str(REMOTE_RUNTIME / "experiment_manifest.yaml"),
            copy=True,
        )
        .add_local_dir(CANDIDATE_ROOT, str(REMOTE_CANDIDATES), copy=True)
        .env(
            {
                "PYTHONPATH": str(REMOTE_RUNTIME),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
    )
else:
    # Function dependencies are already fixed by the hydrated Modal definition.
    # Remote module import must not attempt to resolve the Mac's local sources.
    image = modal.Image.debian_slim(python_version="3.12")

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _candidate_manifest() -> dict[str, Any]:
    return json.loads((CANDIDATE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _remote_candidate_manifest() -> dict[str, Any]:
    return json.loads((REMOTE_CANDIDATES / "manifest.json").read_text(encoding="utf-8"))


def _validate_id(value: str, label: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_SAFE_ID.pattern}")
    return value


def _profile(stage: str):
    from common.training_config import TrainingProfile

    spec = STAGES[stage]
    return TrainingProfile(
        name=f"standalone_{stage}_{spec['steps']}_cuda_v1",
        version="2",
        max_steps=int(spec["steps"]),
        global_batch_size=int(spec["global_batch_size"]),
        microbatch_size=None,
        gradient_accumulation_steps=1,
        peak_learning_rate=0.001,
        adamw_betas=(0.9, 0.98),
        weight_decay=0.1,
        warmup_steps=int(spec["warmup_steps"]),
        scheduler="cosine_decay_to_zero",
        gradient_clip_norm=1.0,
        validation_interval=int(spec["validation_interval"]),
        validation_examples=int(spec["validation_examples"]),
        checkpoint_interval=int(spec["checkpoint_interval"]),
        maximum_wall_seconds=int(spec["maximum_wall_seconds"]),
        dtype="float32",
        deterministic_algorithms=True,
        device_requirement="cuda",
        accelerator_memory_fraction=None,
        scientific=bool(spec["scientific"]),
        cublas_workspace_config=":4096:8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_metrics(
    candidate: Path,
    output: Path,
    seed: int,
    case_count: int,
) -> dict[str, Any]:
    import torch
    from common.candidate_artifact import build_candidate_artifact
    from common.task_adapter import DEFAULT_TASK
    from common.training_config import TrainingSeedBundle
    from common.training_data import public_development_cases

    seeds = TrainingSeedBundle.from_run_seed(seed)
    built = build_candidate_artifact(candidate, seed=seeds.model_initialization_seed)
    model = built.model.to(torch.device("cuda:0"))
    checkpoint = torch.load(
        output / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    cases = public_development_cases(seeds.development_set_seed, case_count)
    correct_teacher_tokens = 0
    total_teacher_tokens = 0
    correct_generated_tokens = 0
    total_generated_tokens = 0
    exact = 0
    batch_size = min(512, case_count)
    with torch.no_grad():
        for offset in range(0, len(cases), batch_size):
            batch = cases[offset : offset + batch_size]
            input_ids, labels = DEFAULT_TASK.collate(batch)
            input_ids = input_ids.to("cuda:0")
            labels = labels.to("cuda:0")
            logits = model(input_ids)
            predictions = logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            mask = targets.ne(-100)
            correct_teacher_tokens += int(
                predictions.eq(targets).logical_and(mask).sum()
            )
            total_teacher_tokens += int(mask.sum())

            prompts = torch.tensor(
                [DEFAULT_TASK.encode_prompt(a, b) for a, b in batch],
                dtype=torch.long,
                device="cuda:0",
            )
            generated = DEFAULT_TASK.generate(model, prompts)
            completions = generated[:, prompts.shape[1] :]
            expected = torch.tensor(
                [
                    DEFAULT_TASK.encode_text(DEFAULT_TASK.target_text(a, b))
                    + [DEFAULT_TASK.vocabulary.index("<eos>")]
                    for a, b in batch
                ],
                dtype=torch.long,
                device="cuda:0",
            )
            correct_generated_tokens += int(completions.eq(expected).sum())
            total_generated_tokens += int(expected.numel())
            exact += int(completions.eq(expected).all(dim=1).sum())
    del model
    torch.cuda.empty_cache()
    return {
        "case_count": case_count,
        "teacher_forced_token_accuracy": (
            correct_teacher_tokens / total_teacher_tokens
            if total_teacher_tokens
            else 0.0
        ),
        "autoregressive_token_accuracy": (
            correct_generated_tokens / total_generated_tokens
            if total_generated_tokens
            else 0.0
        ),
        "autoregressive_exact_match_accuracy": exact / len(cases) if cases else 0.0,
    }


def _official_adderboard_evaluation(
    candidate: Path,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    """Run the vendored AdderBoard verifier without changing its test loop."""

    import contextlib
    import importlib.util
    import io

    import torch
    from common.candidate_artifact import build_candidate_artifact
    from common.task_adapter import DEFAULT_TASK
    from common.training_config import TrainingSeedBundle

    verifier_path = REMOTE_ADDERBOARD / "verify.py"
    verifier_spec = importlib.util.spec_from_file_location(
        "official_adderboard_verify", verifier_path
    )
    if verifier_spec is None or verifier_spec.loader is None:
        raise RuntimeError("official AdderBoard verifier could not be imported")
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)

    seeds = TrainingSeedBundle.from_run_seed(seed)
    built = build_candidate_artifact(candidate, seed=seeds.model_initialization_seed)
    model = built.model.to(torch.device("cuda:0"))
    checkpoint = torch.load(
        output / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    class SubmissionAdapter:
        model: Any = None

        @classmethod
        def build_model(cls):
            return cls.model, {
                "name": candidate.stem,
                "author": "RL4RL research trajectory",
                "params": sum(
                    parameter.numel() for parameter in cls.model.parameters()
                ),
                "architecture": built.metadata.get("architecture", "Architecture IR"),
                "tricks": list(built.metadata.get("tricks", [])),
            }

        @staticmethod
        @torch.no_grad()
        def add(active_model, a: int, b: int) -> int:
            prompts = torch.tensor(
                [DEFAULT_TASK.encode_prompt(a, b)],
                dtype=torch.long,
                device="cuda:0",
            )
            generated = DEFAULT_TASK.generate(active_model, prompts)
            completion = generated[0, prompts.shape[1] :].detach().cpu().tolist()
            return DEFAULT_TASK.decode_generated(completion)

    SubmissionAdapter.model = model
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = verifier.run_test(SubmissionAdapter, num_tests=10_000, seed=2025)
    log = captured.getvalue()
    (output / "official_adderboard_verify.log").write_text(log, encoding="utf-8")
    del SubmissionAdapter.model
    del model
    torch.cuda.empty_cache()
    return {
        "protocol": {
            "source": "anadim/AdderBoard verify.py",
            "verifyPySha256": _sha256(verifier_path),
            "randomSeed": 2025,
            "edgeCaseCount": 10,
            "randomCaseCount": 10_000,
            "totalCaseCount": 10_010,
            "comparison": "integer equality",
            "qualificationThresholdPercent": 99.0,
            "verifierLoopModified": False,
        },
        "passed": int(result["passed"]),
        "total": int(result["total"]),
        "accuracyPercent": float(result["accuracy"]),
        "qualified": bool(result["qualified"]),
        "elapsedSeconds": float(result["time"]),
        "metadata": dict(result["metadata"]),
        "logFile": "official_adderboard_verify.log",
    }


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,
    memory=4096,
    volumes={str(RESULT_ROOT): results},
    timeout=FUNCTION_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=MAX_PARALLEL_JOBS,
    retries=0,
    single_use_containers=True,
    block_network=True,
)
def train_one(
    cohort_id: str,
    candidate_id: str,
    stage: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from common.trainer import train_candidate_in_process
    from common.training_config import TrainingSeedBundle

    _validate_id(cohort_id, "cohort_id")
    _validate_id(candidate_id, "candidate_id")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    manifest = _remote_candidate_manifest()
    matches = [
        item for item in manifest["candidates"] if item["candidateId"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("candidate is not bound by the standalone manifest")
    selected = matches[0]
    candidate = REMOTE_CANDIDATES / selected["filename"]
    if _sha256(candidate) != selected["sha256"]:
        raise RuntimeError("candidate snapshot digest changed")

    profile = _profile(stage)
    output = RESULT_ROOT / cohort_id / candidate_id / f"{stage}-seed-{seed}"
    if output.exists() and any(output.iterdir()):
        summary = output / "standalone_summary.json"
        if summary.is_file():
            return json.loads(summary.read_text(encoding="utf-8"))
        raise RuntimeError(f"nonempty incomplete output already exists: {output}")

    torch.set_num_threads(2)
    training = train_candidate_in_process(
        candidate_path=candidate,
        output_dir=output,
        profile=profile,
        seeds=TrainingSeedBundle.from_run_seed(seed),
        requested_device="cuda",
        allow_cpu_for_tests=False,
    )
    dense = (
        _dense_metrics(
            candidate,
            output,
            seed,
            case_count=int(STAGES[stage]["validation_examples"]),
        )
        if training.success
        else None
    )
    official_adderboard = (
        _official_adderboard_evaluation(candidate, output, seed)
        if training.success
        else None
    )
    summary = {
        "schemaVersion": "1.0",
        "cohortId": cohort_id,
        "candidateId": candidate_id,
        "architectureHash": selected["architectureHash"],
        "parameterCount": selected["parameterCount"],
        "trajectoryIds": selected["trajectoryIds"],
        "stage": stage,
        "seed": seed,
        "profile": asdict(profile),
        "training": training.to_dict(),
        "denseMetrics": dense,
        "officialAdderBoard": official_adderboard,
    }
    (output / "standalone_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results.commit()
    return summary


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    volumes={str(RESULT_ROOT): results},
    timeout=120,
    block_network=True,
)
def cohort_status(cohort_id: str) -> dict[str, Any]:
    _validate_id(cohort_id, "cohort_id")
    results.reload()
    root = RESULT_ROOT / cohort_id
    summaries: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("*/*/standalone_summary.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "cohortId": cohort_id,
        "completed": len(summaries),
        "summaries": summaries,
    }


def _stage_estimate(stage: str, job_count: int) -> dict[str, Any]:
    training_seconds = int(STAGES[stage]["maximum_wall_seconds"])
    # The entire GPU function, including exact official verification, is
    # bounded by the Modal function timeout.
    seconds = FUNCTION_TIMEOUT_SECONDS
    # Current published rates as of 2026-08-24. This is an authorization aid,
    # not a Modal billing limit; actual usage is billed by elapsed resources.
    t4_per_second = 0.000164
    cpu_core_per_second = 0.0000131
    memory_gib_per_second = 0.00000222
    per_job = seconds * (
        t4_per_second + 2 * cpu_core_per_second + 4 * memory_gib_per_second
    )
    return {
        "jobs": job_count,
        "stage": stage,
        "stepsPerJob": int(STAGES[stage]["steps"]),
        "trainingWallTimeCapSecondsPerJob": training_seconds,
        "wallTimeCapSecondsPerJob": seconds,
        "worstCaseComputeEstimateUsd": round(job_count * per_job, 6),
        "estimateIncludesImageBuildOrStorage": False,
    }


@app.local_entrypoint()
def main(
    action: str = "plan",
    stage: str = "develop",
    cohort_id: str = "small-arch-develop-v1",
    seed: int = 1,
    approved: bool = False,
) -> None:
    manifest = _candidate_manifest()
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {sorted(STAGES)}")
    _validate_id(cohort_id, "cohort_id")
    candidate_ids = [item["candidateId"] for item in manifest["candidates"]]

    if action == "plan":
        print(
            json.dumps(
                {
                    "app": APP_NAME,
                    "volume": VOLUME_NAME,
                    "trajectoryCount": manifest["trajectoryCount"],
                    "uniqueCandidateCount": len(candidate_ids),
                    "selectedStage": STAGES[stage],
                    "costEstimate": _stage_estimate(stage, len(candidate_ids)),
                    "candidateIds": candidate_ids,
                    "remoteJobsStarted": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "launch":
        if not approved or os.environ.get("RL4RL_RETRAIN_APPROVED") != "YES":
            raise SystemExit(
                "launch refused: pass --approved and set RL4RL_RETRAIN_APPROVED=YES"
            )
        inputs = [
            (cohort_id, candidate_id, stage, seed) for candidate_id in candidate_ids
        ]
        outcomes = list(
            train_one.starmap(
                inputs,
                order_outputs=True,
                return_exceptions=True,
            )
        )
        completed = []
        failures = []
        for candidate_id, outcome in zip(candidate_ids, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(
                    {
                        "candidateId": candidate_id,
                        "errorType": type(outcome).__name__,
                        "error": str(outcome),
                    }
                )
            else:
                completed.append(
                    {
                        "candidateId": candidate_id,
                        "trainingSuccess": bool(outcome["training"]["success"]),
                        "stepsCompleted": int(outcome["training"]["steps_completed"]),
                        "officialAdderBoardAccuracyPercent": float(
                            outcome["officialAdderBoard"]["accuracyPercent"]
                        ),
                        "officialAdderBoardQualified": bool(
                            outcome["officialAdderBoard"]["qualified"]
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "cohortId": cohort_id,
                    "stage": stage,
                    "seed": seed,
                    "submitted": len(inputs),
                    "completed": completed,
                    "exceptions": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "status":
        print(json.dumps(cohort_status.remote(cohort_id), indent=2, sort_keys=True))
        return
    raise SystemExit("--action must be plan, launch, or status")
