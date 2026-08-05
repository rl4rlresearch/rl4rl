"""Run a bounded, explicitly non-scientific C0-C3 integration study.

This entrypoint exists to exercise provider, training, evaluation, scheduling,
resume, and artifact plumbing while the scientific readiness gates remain
closed. Its outputs are development diagnostics and must never be pooled with
pilot or main-study data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    RunArtifactStore,
)
from common.evaluation_profiles import (
    EvaluationLayer,
    get_evaluation_profile,
    resolve_evaluation_plan,
)
from common.gpt56_sol import TARGET_MODEL, GPT56SolProfile
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.training_config import get_training_profile
from study.budget import BudgetSpec
from study.contracts import StudySpec
from study.randomization import generate_plan, load_or_create_plan
from study.runtime_adapters import (
    CandidateSourceStore,
    LayerACandidateEvaluator,
    MatchedCausalProposalGenerator,
)
from study.scheduling import NoPendingRuns, SequentialRunScheduler
from study.serialization import (
    content_hash,
    create_json_exclusive,
    read_json,
    source_sha256,
)

DEFAULT_API_BASE = "https://api.openai.com/v1"
MAX_DEVELOPMENT_PROVIDER_CALLS = 32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cost-capped C0-C3 integration run. Development diagnostics only; "
            "never scientific evidence."
        )
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--initial-candidate",
        type=Path,
        default=ROOT / "common" / "initial_candidate.py",
    )
    parser.add_argument("--study-seed", type=int, default=20260804)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--opportunities", type=int, default=1)
    parser.add_argument("--portfolio-size", type=int, default=2)
    parser.add_argument("--transition-opportunities", default="1")
    parser.add_argument(
        "--training-profile",
        choices=("smoke_train_v1", "development_train_v1"),
        default="smoke_train_v1",
    )
    parser.add_argument(
        "--evaluation-profile",
        choices=("smoke_eval_v1", "development_eval_v1"),
        default="smoke_eval_v1",
    )
    parser.add_argument("--evaluation-cases", type=int)
    parser.add_argument("--eligibility-threshold", type=float, default=0.0)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=1_200)
    parser.add_argument("--max-prompt-tokens-per-run", type=int, default=20_000)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--max-provider-calls", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-paid-run", action="store_true")
    return parser


def _transition_schedule(text: str, opportunities: int) -> tuple[int, ...]:
    try:
        schedule = tuple(
            sorted({int(value.strip()) for value in text.split(",") if value.strip()})
        )
    except ValueError as error:
        raise ValueError(
            "transition opportunities must be comma-separated integers"
        ) from error
    if opportunities > 0 and not schedule:
        raise ValueError("a non-empty transition schedule is required")
    if any(value < 1 or value > opportunities for value in schedule):
        raise ValueError("transition opportunity outside the proposal budget")
    return schedule


def _code_hash() -> str:
    included = [
        *ROOT.glob("**/*.py"),
        *ROOT.glob("common/prompts/*.md"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    payload = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(included))
        if path.is_file() and ".venv" not in path.parts
    }
    return content_hash(payload)


def _environment_payload(device: str, allow_cpu_for_tests: bool) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "requested_device": device,
        "allow_cpu_for_tests": allow_cpu_for_tests,
        "pytorch_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    }


def _resolve_run(
    arguments: argparse.Namespace,
) -> tuple[StudySpec, dict[str, object]]:
    if arguments.blocks < 1:
        raise ValueError("blocks must be positive")
    if arguments.opportunities < 1:
        raise ValueError("opportunities must be positive")
    if arguments.portfolio_size < 2:
        raise ValueError("portfolio size must be at least two")
    if not 0.0 <= arguments.eligibility_threshold <= 1.0:
        raise ValueError("eligibility threshold must be in [0, 1]")
    if not 128 <= arguments.max_completion_tokens <= 4_096:
        raise ValueError("max completion tokens must be between 128 and 4096")
    if arguments.max_prompt_tokens_per_run < 1:
        raise ValueError("max prompt tokens per run must be positive")
    if arguments.request_timeout_seconds < 1:
        raise ValueError("request timeout must be positive")
    if not 1 <= arguments.max_provider_calls <= MAX_DEVELOPMENT_PROVIDER_CALLS:
        raise ValueError(
            f"max provider calls must be between 1 and {MAX_DEVELOPMENT_PROVIDER_CALLS}"
        )
    if arguments.device == "cpu" and not arguments.allow_cpu_for_tests:
        raise ValueError("CPU development training requires --allow-cpu-for-tests")

    required_provider_calls = arguments.blocks * 4 * arguments.opportunities
    if required_provider_calls > arguments.max_provider_calls:
        raise ValueError(
            "requested design can make "
            f"{required_provider_calls} provider calls, above --max-provider-calls "
            f"{arguments.max_provider_calls}"
        )

    training_profile = get_training_profile(arguments.training_profile)
    evaluation_profile = get_evaluation_profile(arguments.evaluation_profile)
    if training_profile.scientific or evaluation_profile.scientific:
        raise ValueError("development runner refuses scientific profiles")
    evaluation_plan = resolve_evaluation_plan(
        evaluation_profile.name,
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=arguments.evaluation_cases,
    )
    schedule = _transition_schedule(
        arguments.transition_opportunities, arguments.opportunities
    )
    source_path = arguments.initial_candidate.resolve()
    initial_source = source_path.read_text(encoding="utf-8")
    candidate_evaluations_per_run = arguments.opportunities + 1
    budget = BudgetSpec(
        proposal_opportunities=arguments.opportunities,
        provider_attempts_per_opportunity=1,
        prompt_tokens=arguments.max_prompt_tokens_per_run,
        completion_tokens=(arguments.opportunities * arguments.max_completion_tokens),
        repairs=0,
        candidate_training_attempts=candidate_evaluations_per_run,
        training_steps=(candidate_evaluations_per_run * training_profile.max_steps),
        training_examples=(
            candidate_evaluations_per_run
            * training_profile.max_steps
            * training_profile.global_batch_size
        ),
        mps_seconds=(
            candidate_evaluations_per_run * training_profile.maximum_wall_seconds
        ),
        evaluation_cases=(candidate_evaluations_per_run * evaluation_plan.case_count),
        infrastructure_retries=0,
        repair_attempts_per_opportunity=0,
    )
    environment = _environment_payload(arguments.device, arguments.allow_cpu_for_tests)
    common_configuration = {
        "mode": "development_integration",
        "scientific": False,
        "training_profile": training_profile.name,
        "training_profile_hash": training_profile.profile_hash,
        "evaluation_profile": evaluation_profile.name,
        "evaluation_profile_hash": evaluation_profile.profile_hash,
        "evaluation_cases": evaluation_plan.case_count,
        "eligibility_threshold": arguments.eligibility_threshold,
        "device": arguments.device,
        "allow_cpu_for_tests": arguments.allow_cpu_for_tests,
        "generation": {
            "model": TARGET_MODEL,
            "reasoning_effort": arguments.reasoning_effort,
            "max_completion_tokens": arguments.max_completion_tokens,
            "request_timeout_seconds": arguments.request_timeout_seconds,
            "provider_attempts_per_opportunity": 1,
            "repairs": 0,
        },
    }
    spec = StudySpec(
        study_id=arguments.study_id,
        study_seed=arguments.study_seed,
        block_count=arguments.blocks,
        budget=budget,
        portfolio_size=arguments.portfolio_size,
        transition_opportunities=schedule,
        initial_candidate_id=source_sha256(initial_source),
        common_config_hash=content_hash(common_configuration),
        code_hash=_code_hash(),
        environment_hash=content_hash(environment),
        scientific=False,
    )
    maximum_completion_tokens = (
        required_provider_calls * arguments.max_completion_tokens
    )
    manifest = {
        "schema_name": "DevelopmentStudyManifest",
        "schema_version": "1.0",
        "mode": "development_integration",
        "scientific": False,
        "study_id": spec.study_id,
        "study_spec": spec.to_dict(),
        "initial_candidate": str(source_path),
        "configuration": common_configuration,
        "environment": environment,
        "cost_guard": {
            "maximum_provider_calls": required_provider_calls,
            "operator_confirmed_call_ceiling": arguments.max_provider_calls,
            "maximum_completion_tokens_across_study": maximum_completion_tokens,
            "maximum_prompt_tokens_per_run": arguments.max_prompt_tokens_per_run,
            "automatic_provider_retries": 0,
            "format_repairs": 0,
            "usd_ceiling": None,
            "usd_note": (
                "Token/request ceilings are enforced; no USD ceiling is claimed "
                "because price is provider-account specific."
            ),
        },
        "interpretation": {
            "allowed": [
                "provider/training/evaluation/artifact integration diagnostics",
                "candidate-format and infrastructure failure discovery",
            ],
            "forbidden": [
                "scientific pilot or main-study inference",
                "pooling with protocol data",
                "memory-factor inference when only one opportunity is run",
                "architecture ranking from smoke_train_v1",
            ],
        },
    }
    return spec, manifest


def _write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"development manifest changed across resume: {path}")
    else:
        create_json_exclusive(path, payload)


def _api_configuration() -> tuple[str, str, str]:
    if os.environ.get("DISCOVERY_API_KEY"):
        key_name = "DISCOVERY_API_KEY"
    elif os.environ.get("OPENAI_API_KEY"):
        key_name = "OPENAI_API_KEY"
    else:
        raise SystemExit("missing DISCOVERY_API_KEY or OPENAI_API_KEY")
    api_base = os.environ.get("DISCOVERY_API_BASE", DEFAULT_API_BASE)
    return os.environ[key_name], api_base, key_name


def _shared_system_prompt() -> str:
    return "\n\n".join(
        (
            (ROOT / "common" / "prompts" / "shared_system.md").read_text(),
            (ROOT / "common" / "prompts" / "shared_task.md").read_text(),
        )
    )


def _execute(
    arguments: argparse.Namespace, spec: StudySpec, manifest: dict[str, object]
) -> dict[str, object]:
    output_root = arguments.output_root.resolve()
    study_root = output_root / spec.study_id
    _write_once(study_root / "development_manifest.json", manifest)
    plan = load_or_create_plan(
        spec,
        output_root=output_root,
        plan_path=study_root / "randomization_plan.json",
    )
    scheduler = SequentialRunScheduler(
        plan,
        state_path=study_root / "schedule_state.json",
        lease_path=output_root / ".study_mps.lock",
    )
    api_key, api_base, key_name = _api_configuration()
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        max_retries=0,
        timeout=arguments.request_timeout_seconds,
    )
    initial_source = arguments.initial_candidate.resolve().read_text(encoding="utf-8")
    shared_system = _shared_system_prompt()

    while True:
        try:
            claim = scheduler.claim_next()
        except NoPendingRuns:
            break
        with claim as run:
            run_root = Path(run.run_directory)
            source_store = CandidateSourceStore(run_root / "candidate_sources")
            source_store.register(spec.initial_candidate_id, initial_source)
            generation = GPT56SolProfile.resolve(
                model=TARGET_MODEL,
                seed=run.run_seed,
                default_reasoning_effort=arguments.reasoning_effort,
                default_max_completion_tokens=arguments.max_completion_tokens,
                default_timeout_seconds=arguments.request_timeout_seconds,
                default_retries=0,
                default_retry_delay_seconds=0,
                environ={},
            )
            generator = MatchedCausalProposalGenerator(
                client=client,
                generation=generation,
                source_store=source_store,
                portfolio_size=spec.portfolio_size,
                system_prompt=shared_system,
                request_log_root=run_root / "provider_records",
            )
            evaluator = LayerACandidateEvaluator(
                study_id=spec.study_id,
                block_id=run.block_id,
                run_id=run.run_id,
                condition_id=run.condition.condition_id.value,
                initial_candidate_id=spec.initial_candidate_id,
                source_store=source_store,
                output_root=run_root,
                training_profile=arguments.training_profile,
                device=arguments.device,
                allow_cpu_for_tests=arguments.allow_cpu_for_tests,
                evaluation_profile=arguments.evaluation_profile,
                evaluation_case_count=(
                    get_evaluation_profile(
                        arguments.evaluation_profile
                    ).resolve_case_count(arguments.evaluation_cases)
                ),
                pi_decision_record_id=None,
                eligibility_threshold=arguments.eligibility_threshold,
            )
            artifact_store = RunArtifactStore(
                run_root / "artifact_ledger",
                ArtifactContext(
                    study_id=spec.study_id,
                    block_id=run.block_id,
                    run_id=run.run_id,
                    condition_id=run.condition.condition_id.value,
                    writer_component="scripts.study_development_run",
                    code_sha256=spec.code_hash,
                    config_sha256=spec.common_config_hash,
                    environment_sha256=spec.environment_hash,
                    run_seed=run.run_seed,
                    assignment_sha256=run.assignment_hash,
                ),
            )
            ArtifactEmittingStudyEngine(
                study=spec,
                run=run,
                generator=generator,
                evaluator=evaluator,
                artifact_sink=ImmutableStudyEventSink(artifact_store),
                evaluation_lease_path=None,
            ).execute()

    run_summaries = []
    frozen_indexes = []
    totals = {
        "provider_attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "unknown_provider_usage": 0,
        "candidate_training_attempts": 0,
        "training_steps": 0,
        "training_examples": 0,
        "mps_seconds": 0.0,
        "evaluation_cases": 0,
    }
    for run in plan.runs:
        state = read_json(Path(run.run_directory) / "run_state.json")
        ledger = state["ledger"]
        for name in totals:
            totals[name] += ledger[name]
        store = RunArtifactStore.open(Path(run.run_directory) / "artifact_ledger")
        frozen_index, _ = store.load_frozen_index("search_completion")
        frozen_indexes.append(frozen_index.to_dict())
        run_summaries.append(
            {
                "run_id": run.run_id,
                "condition_id": run.condition.condition_id.value,
                "status": state["status"],
                "accepted": ledger["accepted"],
                "rejected": ledger["rejected"],
                "invalid": ledger["invalid"],
                "scientific_failures": ledger["scientific_failures"],
                "infrastructure_failures": ledger["infrastructure_failures"],
            }
        )
    index_manifest = {
        "schema_name": "StudyArtifactIndexManifest",
        "schema_version": "1.0",
        "study_id": spec.study_id,
        "study_phase": "development",
        "scientific": False,
        "assignment_hash": plan.assignment_hash,
        "run_indexes": frozen_indexes,
    }
    index_manifest_path = study_root / "artifact_index_manifest.json"
    _write_once(index_manifest_path, index_manifest)
    return {
        "mode": "development_integration",
        "scientific": False,
        "study_id": spec.study_id,
        "credential_variable": key_name,
        "api_base": api_base,
        "scheduler": scheduler.summary(),
        "totals": totals,
        "runs": run_summaries,
        "development_manifest": str(study_root / "development_manifest.json"),
        "artifact_index_manifest": str(index_manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec, manifest = _resolve_run(arguments)
    except (OSError, ValueError) as error:
        raise SystemExit(f"development run configuration invalid: {error}") from error
    plan = generate_plan(spec, arguments.output_root)
    preview = {
        **manifest,
        "dry_run": bool(arguments.dry_run),
        "randomization_assignment_hash": plan.assignment_hash,
        "run_order": [
            {
                "run_id": run.run_id,
                "condition_id": run.condition.condition_id.value,
                "order_index": run.order_index,
            }
            for run in plan.runs
        ],
    }
    if arguments.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    print(json.dumps(_execute(arguments, spec, manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
