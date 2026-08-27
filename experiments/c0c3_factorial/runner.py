"""Execute one resumable proposal opportunity through the shared controller."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .agent_scheduler import (
    AgentWorkerLease,
    acquire_agent_worker_slot,
    release_agent_worker_slot,
)
from .artifacts import (
    candidate_hash,
    make_read_only,
    materialize_candidate,
    protected_hash,
    scientific_runtime_hash,
    snapshot_candidate,
)
from .codex_cli import CodexCli, session_id_from_events, usage_from_events
from .evaluator import make_command_evaluator, task_local_evaluator_root
from .fashion_mnist import preflight_candidate_source as preflight_fashion_mnist
from .frameworks import make_framework_adapter
from .native_openevolve import (
    finalize_native_outcome,
    is_native_openevolve,
    prepare_native_selection,
    reset_native_population_from_incumbent,
    stage_native_outcome,
)
from .neutral_task import (
    ARTIFACT_CLEAN_PROMPT_PROFILES,
    FASHION_MNIST_TASK_ADAPTER,
    PAIR_TOKEN_TASK_ADAPTER_V2,
    PAIR_TOKEN_TASK_ADAPTER_V3,
    SUBJECT_NEUTRAL_PROMPT_PROFILES,
)
from .periodic_refresh import (
    REFRESH_INCUMBENT,
    apply_periodic_refresh,
    memory_state,
)
from .prompts import (
    FROZEN_ASSUMPTION_PROMPT,
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    VisibleOutcome,
)
from .spec import (
    EVALUATOR_CONCURRENCY_BY_PROTOCOL,
    ConversationMode,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    TaskSpec,
)
from .state import Evaluation, SearchController
from .task_evaluators import preflight_candidate_source
from .training_ladder import assess_developmental_value, evaluate_training_ladder
from .v3 import load_runtime_options, prompt_renderer_paths
from .v3_analysis import record_candidate_provenance, write_manipulation_packet


class RunLockedError(RuntimeError):
    """Another process already owns this run's mutation boundary."""


@contextmanager
def _run_lock(run_dir: Path):
    path = run_dir / ".runner.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunLockedError(f"run is already active: {run_dir}") from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            handle,
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _register_conversation_session(
    run_dir: Path, session_id: str, *, allow_multiple_for_run: bool = False
) -> None:
    """Atomically prove that one persisted Codex thread belongs to one run."""

    campaign = run_dir.parent.parent
    registry_path = campaign / ".conversation-session-registry.json"
    lock_path = campaign / ".conversation-session-registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            registry = (
                json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.is_file()
                else {}
            )
            if not isinstance(registry, dict):
                raise RuntimeError("conversation session registry is invalid")
            owner = registry.get(session_id)
            if owner is not None and owner != run_dir.name:
                raise RuntimeError(
                    "Codex conversation session is already owned by another run"
                )
            owned_ids = [
                key for key, value in registry.items() if value == run_dir.name
            ]
            if owned_ids and session_id not in owned_ids and not allow_multiple_for_run:
                raise RuntimeError(
                    "run attempted to switch Codex conversation sessions"
                )
            registry[session_id] = run_dir.name
            temporary = registry_path.with_name(
                f".{registry_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(registry, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, registry_path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _hashes(rendered) -> dict[str, str]:
    return {
        "common_template_sha256": rendered.common_template_sha256,
        "search_state_sha256": rendered.search_state_sha256,
        "proposal_policy_sha256": rendered.proposal_policy_sha256,
        "prompt_sha256": rendered.prompt_sha256,
        "treatment_skeleton_sha256": rendered.treatment_skeleton_sha256,
    }


def _opaque_run_id(run_dir: Path) -> str:
    return (
        hashlib.sha256(str(run_dir).encode())
        .hexdigest()
        .translate(str.maketrans({"c": "w", "0": "q", "1": "r", "2": "s", "3": "t"}))[
            :24
        ]
    )


def _refresh_continuous_workspace(
    *,
    run_dir: Path,
    support_source: Path,
    selected_snapshot: Path,
    editable_paths: tuple[str, ...],
    neutral_subject: bool = False,
    artifact_clean_subject: bool = False,
) -> Path:
    """Refresh the stable Codex cwd with this opportunity's selected parent.

    The controller uses one stable, opaque, per-run session workspace, while
    immutable per-opportunity workspaces remain the snapshot/evaluation record.
    Every initial and resumed Codex subprocess is also launched with this path
    as its real operating-system cwd.
    """

    if neutral_subject:
        neutral_root = (
            Path(tempfile.gettempdir()) / "transformer-optimization-workspaces"
        )
        neutral_root.mkdir(parents=True, exist_ok=True)
        opaque_id = _opaque_run_id(run_dir)
        workspace = neutral_root / opaque_id
    else:
        workspace = run_dir / ".continuous-codex-workspace"
    if workspace.exists():
        if not workspace.is_dir() or workspace.is_symlink():
            raise RuntimeError("continuous Codex workspace has an unsafe path type")
        _make_tree_owner_writable(workspace)
        shutil.rmtree(workspace)
    materialize_candidate(
        support_source,
        selected_snapshot,
        workspace,
        editable_paths,
    )
    if artifact_clean_subject:
        _initialize_subject_git_workspace(workspace)
    else:
        identity = hashlib.sha256(f"workspace:{run_dir}".encode()).hexdigest()
        (workspace / ".workspace-identity").write_text(
            identity + "\n", encoding="utf-8"
        )
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("continuous Codex workspace contains a symlink")
    return workspace


def _initialize_subject_git_workspace(workspace: Path) -> None:
    """Provide ordinary local source history without experiment metadata."""

    commands = (
        ("git", "init", "--quiet"),
        ("git", "add", "--all"),
        (
            "git",
            "-c",
            "user.name=Workspace",
            "-c",
            "user.email=workspace@local.invalid",
            "commit",
            "--quiet",
            "-m",
            "Starting source",
        ),
    )
    for command in commands:
        subprocess.run(
            command,
            cwd=workspace,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    exclude = workspace / ".git/info/exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("\n.design-references/\n.subject-cache/\n")


def _make_tree_owner_writable(root: Path) -> None:
    """Allow safe cleanup of controller-created read-only reference trees."""

    paths = [root, *root.rglob("*")]
    for path in reversed(paths):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _recent_outcomes(
    run_dir: Path, *, limit: int = 12, minimum_opportunity: int = 1
) -> tuple[VisibleOutcome, ...]:
    events = run_dir / "events.jsonl"
    if not events.is_file():
        return ()
    records = []
    assessments: dict[int, dict[str, object]] = {}
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
        if record.get("event") == "developmental_assessment" and isinstance(
            record.get("opportunity"), int
        ):
            assessments[int(record["opportunity"])] = record
    outcomes: list[VisibleOutcome] = []
    for record in records:
        if record.get("event") != "proposal_completed":
            continue
        if int(record.get("opportunity", 0)) < minimum_opportunity:
            continue
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        metrics = evaluation.get("metrics")
        opportunity = int(record["opportunity"])
        developmental = assessments.get(opportunity, {})
        outcomes.append(
            VisibleOutcome(
                opportunity=opportunity,
                hypothesis=str(record.get("hypothesis", "[not available]")),
                intended_edit=str(record.get("intended_edit", "[not available]")),
                metrics=dict(metrics) if isinstance(metrics, dict) else {},
                valid=bool(evaluation.get("valid")),
                retained=bool(record.get("retained")),
                failure_kind=(
                    str(evaluation["failure_kind"])
                    if evaluation.get("failure_kind") is not None
                    else None
                ),
                mechanism=str(record.get("mechanism", "[not recorded]")),
                evidence=str(record.get("evidence", "[not recorded]")),
                developmental_status=(
                    str(developmental["status"])
                    if developmental.get("status") is not None
                    else None
                ),
                developmental_credit=(
                    float(developmental["credit"])
                    if isinstance(developmental.get("credit"), int | float)
                    else None
                ),
                developmental_reasons=tuple(
                    str(value) for value in developmental.get("reasons", [])
                ),
            )
        )
    return tuple(outcomes[-limit:])


def _informative_outcomes(
    run_dir: Path,
    *,
    item_limit: int,
    character_limit: int,
    minimum_opportunity: int = 1,
) -> tuple[VisibleOutcome, ...]:
    """Select bounded useful evidence without an LLM or a mechanism menu."""

    all_outcomes = list(
        _recent_outcomes(
            run_dir,
            limit=1000000,
            minimum_opportunity=minimum_opportunity,
        )
    )
    if not all_outcomes:
        return ()
    selected: dict[int, VisibleOutcome] = {
        all_outcomes[-1].opportunity: all_outcomes[-1]
    }
    retained = [
        outcome for outcome in all_outcomes if outcome.retained and outcome.valid
    ]
    if retained:
        selected[retained[-1].opportunity] = retained[-1]
    failures_by_description: dict[str, VisibleOutcome] = {}
    for outcome in reversed(all_outcomes):
        if outcome.valid:
            continue
        provenance = (
            run_dir
            / "opportunities"
            / f"{outcome.opportunity:04d}"
            / "candidate-provenance.json"
        )
        fingerprint = None
        if provenance.is_file():
            value = json.loads(provenance.read_text(encoding="utf-8"))
            fingerprint = value.get("semantic_delta_fingerprint")
        key = str(fingerprint or outcome.failure_kind or "failure").casefold()
        failures_by_description.setdefault(key, outcome)
    for outcome in failures_by_description.values():
        if len(selected) >= item_limit:
            break
        selected[outcome.opportunity] = outcome
    for outcome in reversed(all_outcomes):
        if len(selected) >= item_limit:
            break
        selected.setdefault(outcome.opportunity, outcome)
    ordered = [selected[key] for key in sorted(selected)]
    while ordered and sum(len(repr(outcome)) for outcome in ordered) > character_limit:
        removable = next(
            (
                index
                for index, outcome in enumerate(ordered)
                if outcome is not ordered[-1]
            ),
            None,
        )
        if removable is None:
            break
        ordered.pop(removable)
    return tuple(ordered)


def _mechanism_ledger(
    run_dir: Path, *, limit: int = 24, minimum_opportunity: int = 1
) -> str:
    """Summarize free-form mechanism provenance without replaying raw history."""

    events = run_dir / "events.jsonl"
    if not events.is_file():
        return "No earlier mechanism result is available."
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "proposal_completed":
            continue
        if int(record.get("opportunity", 0)) < minimum_opportunity:
            continue
        label = str(record.get("mechanism", "[not recorded]")).strip()
        key = label.casefold()
        if key not in grouped:
            grouped[key] = {"label": label, "attempts": 0}
            order.append(key)
        summary = grouped[key]
        summary["attempts"] = int(summary["attempts"]) + 1
        evaluation = record.get("evaluation")
        valid = isinstance(evaluation, dict) and bool(evaluation.get("valid"))
        summary["last_result"] = (
            "qualified"
            if valid
            else str(
                evaluation.get("failure_kind", "failed")
                if isinstance(evaluation, dict)
                else "failed"
            )
        )
        if valid and isinstance(evaluation, dict):
            metrics = evaluation.get("metrics")
            if isinstance(metrics, dict) and isinstance(
                metrics.get("parameters"), int | float
            ):
                value = int(metrics["parameters"])
                previous = summary.get("best_parameters")
                summary["best_parameters"] = (
                    value if not isinstance(previous, int) else min(previous, value)
                )
    if not grouped:
        return "No earlier mechanism result is available."
    rows = []
    for key in order[-limit:]:
        value = grouped[key]
        best = value.get("best_parameters", "none")
        rows.append(
            f"- {value['label']}: attempts={value['attempts']}; "
            f"last={value['last_result']}; best_qualified_parameters={best}"
        )
    return "\n".join(rows)


def _copy_editable_files(
    source: Path, destination: Path, editable_paths: tuple[str, ...]
) -> None:
    """Copy only the proposed candidate files into an auditable opportunity."""

    for relative in editable_paths:
        source_file = source / relative
        destination_file = destination / relative
        if not source_file.is_file() or source_file.is_symlink():
            raise ValueError(f"continuous Codex workspace is missing {relative}")
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)


def _run_one_opportunity_unlocked(
    run_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
    allow_v3_prefix_leader: bool = False,
    agent_worker_lease: AgentWorkerLease | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    if (
        spec.conversation_mode is ConversationMode.CONTINUOUS
        and framework.framework_id is not FrameworkKind.AUTORESEARCH
        and spec.protocol_version != "3.0"
    ):
        raise ValueError(
            "continuous Codex sessions currently support Autoresearch only"
        )
    controller = SearchController.load(run_dir, spec)
    if (
        spec.paired_prefix
        and controller.state.next_opportunity < spec.first_fork_opportunity
        and not allow_v3_prefix_leader
    ):
        raise RuntimeError(
            "v3 shared-prefix opportunities must use the paired v3 runner"
        )
    run_manifest = _read_json(run_dir / "manifest.json")
    expected_runtime_hash = scientific_runtime_hash(
        repo_root, task=task, framework=framework
    )
    if (
        spec.protocol_version != "3.0"
        and run_manifest.get("scientific_runtime_hash") != expected_runtime_hash
    ):
        raise ValueError(
            "scientific runtime changed after campaign creation; create a new campaign"
        )
    assignment = run_manifest.get("assignment")
    if (
        not isinstance(assignment, dict)
        or isinstance(assignment.get("run_seed"), bool)
        or not isinstance(assignment.get("run_seed"), int)
    ):
        raise ValueError("run manifest lacks an integer assignment run_seed")
    run_seed = int(assignment["run_seed"])
    if controller.state.active is not None:
        raise RuntimeError(
            "run has an active opportunity; inspect its logs before explicit recovery"
        )
    semantic = run_manifest.get("semantic_intervention")
    state_policy = (
        str(semantic.get("state_policy", "preserve"))
        if isinstance(semantic, dict)
        else "preserve"
    )
    next_opportunity = controller.state.next_opportunity
    if (
        state_policy == REFRESH_INCUMBENT
        and next_opportunity in spec.transition_opportunities
    ):
        refreshed = apply_periodic_refresh(
            run_dir,
            controller=controller,
            base_seed=run_seed,
            opportunity=next_opportunity,
        )
        if is_native_openevolve(framework):
            reset_native_population_from_incumbent(
                run_dir,
                opportunity=next_opportunity,
                search_seed=int(refreshed["search_seed"]),
            )
    subject_memory = memory_state(run_dir, base_seed=run_seed)
    history_start_opportunity = int(
        subject_memory.get("history_start_opportunity", 1)
    )
    search_seed = int(subject_memory.get("search_seed", run_seed))
    native_selection = None
    if is_native_openevolve(framework):
        native_selection = prepare_native_selection(
            run_dir,
            controller=controller,
            task=task,
            framework=framework,
            vendor_root=repo_root / "architecture_discovery/vendor/openevolve",
            run_seed=search_seed,
            opportunity=controller.state.next_opportunity,
        )
        active = controller.begin(
            external_visible_ids=list(native_selection.visible_ids),
            external_parent_id=native_selection.parent_id,
        )
    else:
        active = controller.begin()
    opportunity_root = run_dir / "opportunities" / f"{active.index:04d}"
    opportunity_root.mkdir(parents=True, exist_ok=False)
    support_source = run_dir / "task-support"
    candidate_store = run_dir / "candidates"
    selected_snapshot = candidate_store / active.selected_parent_id
    workspace = opportunity_root / "proposal-workspace"
    materialize_candidate(
        support_source,
        selected_snapshot,
        workspace,
        task.editable_paths,
    )
    neutral_subject = framework.prompt_profile in SUBJECT_NEUTRAL_PROMPT_PROFILES
    artifact_clean_subject = framework.prompt_profile in ARTIFACT_CLEAN_PROMPT_PROFILES
    reference_directory = (
        ".design-references" if neutral_subject else ".factorial-visible"
    )
    visible_root = workspace / reference_directory
    visible_root.mkdir()
    visible_workspaces: list[Path] = []
    visible_candidates: list[VisibleCandidate] = []
    visible_records: list[dict[str, object]] = []
    for index, identifier in enumerate(active.visible_ids, start=1):
        candidate = controller.state.candidates[identifier]
        destination = visible_root / (
            f"design-{index}" if neutral_subject else f"slot-{index}"
        )
        materialize_candidate(
            support_source,
            candidate_store / identifier,
            destination,
            task.editable_paths,
        )
        make_read_only(destination)
        visible_workspaces.append(destination)
        visible_candidates.append(
            VisibleCandidate(
                candidate_id=identifier,
                fitness=candidate.fitness,
                metrics=candidate.metrics,
                selected_count=candidate.selected_count,
                artifact_path=(
                    f"{reference_directory}/design-{index}"
                    if neutral_subject
                    else f"{reference_directory}/slot-{index}"
                ),
                hypothesis=candidate.hypothesis,
            )
        )
        visible_records.append(
            {
                "candidate_id": identifier,
                "metrics": (
                    {
                        name: candidate.metrics[name]
                        for name in task.public_feedback_metrics
                        if name in candidate.metrics
                    }
                    if artifact_clean_subject
                    else candidate.metrics
                ),
                "hypothesis": candidate.hypothesis,
            }
        )
    remaining = controller.remaining()
    after_token_threshold = (
        spec.continues_after_token_threshold
        and controller.state.usage.total_tokens >= spec.budget.max_total_tokens
    )
    show_token_continuation_notice = (
        after_token_threshold
        and not controller.state.token_budget_continuation_notice_sent
    )
    v3_options = (
        load_runtime_options(run_dir.parent.parent)
        if spec.protocol_version == "3.0"
        else None
    )
    if v3_options is not None:
        conversation_options = dict(v3_options.get("conversation", {}))
        recent_outcomes = _informative_outcomes(
            run_dir,
            item_limit=int(conversation_options.get("evidence_item_limit", 10)),
            character_limit=int(
                conversation_options.get("evidence_character_limit", 24000)
            ),
            minimum_opportunity=history_start_opportunity,
        )
    else:
        recent_outcomes = (
            ()
            if controller.state.no_search or not neutral_subject
            else _recent_outcomes(
                run_dir,
                limit=(
                    1
                    if artifact_clean_subject
                    and framework.framework_id is FrameworkKind.AUTORESEARCH
                    else 12
                ),
            )
        )
    session_span = (
        int(conversation_options.get("session_span_opportunities", 1))
        if v3_options is not None
        else 1
    )
    if session_span < 1:
        raise ValueError("session_span_opportunities must be positive")
    phased_session = v3_options is not None and session_span > 1
    semantic_policy: str | None = None
    semantic = run_manifest.get("semantic_intervention")
    if active.transition_active and isinstance(semantic, dict):
        relative = semantic.get("prompt_path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("semantic intervention lacks a prompt path")
        prompt_root = (run_dir.parent.parent / "prompt-bundle").resolve()
        policy_path = (prompt_root / relative).resolve()
        if prompt_root not in policy_path.parents or not policy_path.is_file():
            raise ValueError("semantic intervention prompt path is invalid")
        semantic_policy = policy_path.read_text(encoding="utf-8")
    evaluation_policy_note: str | None = None
    multi_fidelity_options = (
        dict(dict(v3_options.get("evaluation", {})).get("multi_fidelity", {}))
        if v3_options is not None
        else {}
    )
    candidate_ladder = dict(multi_fidelity_options.get("candidate_editable_policy", {}))
    if bool(multi_fidelity_options.get("enabled", False)) and bool(
        candidate_ladder.get("enabled", False)
    ):
        levels_symbol = str(candidate_ladder.get("levels_symbol", "EVALUATION_LADDER"))
        thresholds_symbol = candidate_ladder.get("thresholds_symbol")
        evaluation_policy_note = (
            "Verification uses a multi-fidelity training ladder. You may edit the "
            f"literal {levels_symbol} policy in "
            f"{candidate_ladder.get('path')}"
        )
        if thresholds_symbol:
            evaluation_policy_note += f" and its literal {thresholds_symbol} policy"
        evaluation_policy_note += (
            ". The verifier bounds the rung count and range, guarantees the common "
            "terminal budget, and applies the same official success and retention "
            "rules regardless of the intermediate ladder."
        )
    prompt_context = PromptContext(
        condition=controller.condition,
        opportunity=active.index,
        selected_parent_id=active.selected_parent_id,
        visible_candidates=tuple(visible_candidates),
        remaining_proposals=int(remaining["proposals"]),
        remaining_evaluations=int(remaining["evaluations"]),
        remaining_tokens=int(remaining["tokens"]),
        remaining_evaluator_seconds=float(remaining["evaluator_seconds"]),
        hide_token_budget=after_token_threshold,
        token_budget_continuation_notice=show_token_continuation_notice,
        no_search=controller.state.no_search,
        recent_outcomes=recent_outcomes,
        mechanism_ledger=(
            _mechanism_ledger(
                run_dir,
                minimum_opportunity=history_start_opportunity,
            )
            if framework.adapter.startswith("controlled_openevolve_prompt_diff_")
            else "No earlier mechanism result is available."
        ),
        phased_session=phased_session,
        proposal_policy_override=semantic_policy,
        evaluation_policy_note=evaluation_policy_note,
    )
    frozen_transition = run_dir / FROZEN_ASSUMPTION_PROMPT
    if (
        framework.prompt_profile in ARTIFACT_CLEAN_PROMPT_PROFILES
        and spec.protocol_version != "3.0"
        and not frozen_transition.is_file()
    ):
        raise RuntimeError(
            "artifact-clean trajectory lacks its start-time assumption prompt snapshot"
        )
    if spec.protocol_version == "3.0":
        template_root, transition_override = prompt_renderer_paths(
            run_dir.parent.parent, spec=spec, framework=framework
        )
    else:
        template_root = repo_root / "experiments/c0c3_factorial/templates"
        transition_override = frozen_transition if frozen_transition.is_file() else None
    renderer = PromptRenderer(
        template_root,
        artifact_clean_transition_override=transition_override,
    )
    rendered = renderer.render(spec, task, framework, prompt_context)
    (opportunity_root / "prompt.md").write_text(rendered.text, encoding="utf-8")
    (opportunity_root / "prompt_manifest.json").write_text(
        json.dumps(_hashes(rendered), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if spec.protocol_version == "3.0":
        (opportunity_root / "state-capsule.json").write_text(
            json.dumps(
                {
                    "schema_version": "3.0",
                    "selected_source_sha256": active.selected_parent_id,
                    "public_evidence": [
                        {
                            "metrics": record["metrics"],
                            "hypothesis": record["hypothesis"],
                        }
                        for record in visible_records
                    ],
                    "evidence_opportunities": [
                        outcome.opportunity for outcome in recent_outcomes
                    ],
                    "original_outcome_count": max(
                        0,
                        controller.state.proposals_used
                        - history_start_opportunity
                        + 1,
                    ),
                    "rendered_outcome_count": len(recent_outcomes),
                    "rendered_evidence_characters": sum(
                        len(repr(outcome)) for outcome in recent_outcomes
                    ),
                    "visible_branch_count": len(visible_candidates),
                    "rendered_prompt_characters": len(rendered.text),
                    "rendered_prompt_sha256": rendered.prompt_sha256,
                    "runtime_options_sha256": hashlib.sha256(
                        json.dumps(
                            v3_options, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if show_token_continuation_notice:
        controller.record_token_budget_continuation_notice()
    continuous = spec.conversation_mode is ConversationMode.CONTINUOUS or phased_session
    codex_workspace = (
        _refresh_continuous_workspace(
            run_dir=run_dir,
            support_source=support_source,
            selected_snapshot=selected_snapshot,
            editable_paths=task.editable_paths,
            neutral_subject=neutral_subject,
            artifact_clean_subject=artifact_clean_subject,
        )
        if continuous
        else workspace
    )
    if continuous and neutral_subject:
        reference_target = codex_workspace / reference_directory
        shutil.copytree(visible_root, reference_target)
        make_read_only(reference_target)
    before_protected = protected_hash(codex_workspace, task.editable_paths)
    parent_hash = candidate_hash(workspace, task.editable_paths)
    adapter = make_framework_adapter(
        framework,
        CodexCli(codex_binary),
        repo_root=repo_root,
        prompt_template_root=(
            template_root if spec.protocol_version == "3.0" else None
        ),
    )
    if phased_session and active.index > 1 and (active.index - 1) % session_span == 0:
        controller.reset_conversation_session(
            opportunity=active.index,
            reason=f"configured {session_span}-opportunity research phase ended",
        )
    requested_session_id = (
        controller.state.conversation_session_id if continuous else None
    )
    try:
        proposal = adapter.propose(
            rendered=rendered,
            workspace=codex_workspace,
            model=spec.model,
            log_root=opportunity_root / "codex",
            call_id=f"proposal-{active.index}",
            timeout_seconds=codex_timeout_seconds,
            task=task,
            visible_workspaces=tuple(visible_workspaces),
            selected_parent_id=active.selected_parent_id,
            visible_records=tuple(visible_records),
            run_seed=search_seed,
            resume_session_id=requested_session_id,
            persist_session=continuous,
            neutral_subject=neutral_subject,
            artifact_clean_subject=artifact_clean_subject,
            native_prompt_context=(
                native_selection.prompt_context()
                if native_selection is not None
                else None
            ),
        )
    finally:
        # The thirty-worker ceiling controls concurrent subject agents. Local
        # and remote evaluators use their independent task/host schedulers.
        release_agent_worker_slot(agent_worker_lease)
    protected_changed = (
        protected_hash(codex_workspace, task.editable_paths) != before_protected
    )
    adapter_error = proposal.adapter_error
    if continuous:
        if proposal.codex.session_id is None:
            if proposal.codex.returncode == 0:
                adapter_error = adapter_error or (
                    "Codex did not record a resumable session ID"
                )
        elif (
            requested_session_id is not None
            and proposal.codex.session_id != requested_session_id
        ):
            adapter_error = adapter_error or (
                "Codex resumed a different conversation session"
            )
        else:
            try:
                _register_conversation_session(
                    run_dir,
                    proposal.codex.session_id,
                    allow_multiple_for_run=phased_session,
                )
                controller.record_conversation_session(proposal.codex.session_id)
            except (OSError, RuntimeError, ValueError) as error:
                adapter_error = adapter_error or str(error)
    if continuous:
        try:
            _copy_editable_files(codex_workspace, workspace, task.editable_paths)
        except (OSError, ValueError) as error:
            adapter_error = adapter_error or str(error)
    try:
        child_hash = candidate_hash(workspace, task.editable_paths)
    except (OSError, ValueError) as error:
        child_hash = hashlib.sha256(
            f"invalid:{controller.state.run_id}:{active.index}".encode()
        ).hexdigest()
        adapter_error = adapter_error or str(error)
    if child_hash == parent_hash:
        adapter_error = adapter_error or "proposal made no editable candidate change"
    if protected_changed:
        adapter_error = adapter_error or "proposal modified protected task files"
    candidate_id, candidate_snapshot = snapshot_candidate(
        workspace, candidate_store, task.editable_paths
    )
    if candidate_id != child_hash:
        raise RuntimeError("candidate changed while being snapshotted")
    recorded_candidate_id = candidate_id
    if candidate_id in controller.state.candidates:
        adapter_error = adapter_error or "proposal reproduced an evaluated candidate"
        recorded_candidate_id = hashlib.sha256(
            (
                f"duplicate:{candidate_id}:{controller.state.run_id}:{active.index}"
            ).encode()
        ).hexdigest()
    preflight_error = None
    if (
        proposal.codex.returncode == 0
        and adapter_error is None
        and task.adapter in {PAIR_TOKEN_TASK_ADAPTER_V2, PAIR_TOKEN_TASK_ADAPTER_V3}
    ):
        preflight_error = preflight_candidate_source(workspace)
    elif (
        proposal.codex.returncode == 0
        and adapter_error is None
        and task.adapter == FASHION_MNIST_TASK_ADAPTER
    ):
        preflight_error = preflight_fashion_mnist(workspace)
    if proposal.codex.returncode != 0 or adapter_error or preflight_error:
        failure_kind = (
            "provider"
            if proposal.codex.returncode != 0
            else (
                "source_preflight"
                if preflight_error is not None
                else proposal.adapter_failure_kind or "invalid_candidate"
            )
        )
        evaluation = Evaluation(
            valid=False,
            fitness=None,
            metrics={
                "codex_returncode": proposal.codex.returncode,
                "adapter_error": adapter_error,
                "preflight_error": preflight_error,
                "protected_changed": protected_changed,
            },
            evaluator_seconds=0.0,
            evaluator_calls=0,
            failure_kind=failure_kind,
        )
    else:
        max_parallel_evaluators = (
            int(dict(v3_options.get("evaluation", {})).get("task_pool_capacity", 1))
            if v3_options is not None
            else (
                spec.blocks
                if spec.protocol_version in {"1.7", "2.1"}
                else EVALUATOR_CONCURRENCY_BY_PROTOCOL.get(spec.protocol_version)
            )
        )
        slot_root = (
            (
                task_local_evaluator_root(task.task_id)
                if spec.protocol_version == "3.0"
                else run_dir.parent.parent / ".evaluator-slots" / "campaign"
            )
            if max_parallel_evaluators is not None
            else None
        )

        def evaluator_for(stage_task: TaskSpec):
            return make_command_evaluator(
                task=stage_task,
                support_source=support_source,
                repo_root=repo_root,
                python_bin=python_bin,
                slot_root=slot_root,
                max_parallel_evaluators=max_parallel_evaluators,
            )

        replicate_count = (
            int(dict(v3_options.get("evaluation", {})).get("paired_training_seeds", 1))
            if v3_options is not None
            else 1
        )
        if replicate_count < 1:
            raise ValueError("paired_training_seeds must be positive")
        multi_fidelity = multi_fidelity_options
        controller_managed_ladder = bool(
            multi_fidelity.get("enabled", False)
        ) and not str(multi_fidelity.get("strategy", "")).startswith("in_process_")
        if controller_managed_ladder:
            if replicate_count != 1:
                raise ValueError(
                    "training ladders and repeated training seeds cannot be combined "
                    "in one online evaluator call"
                )
            evaluation = evaluate_training_ladder(
                task=task,
                config=multi_fidelity,
                candidate_snapshot=candidate_snapshot,
                opportunity_root=opportunity_root,
                timeout_seconds=spec.budget.evaluator_timeout_seconds,
                run_seed=run_seed,
                evaluator_factory=evaluator_for,
            )
        elif replicate_count == 1:
            evaluated = evaluator_for(task).evaluate(
                candidate_snapshot=candidate_snapshot,
                opportunity_root=opportunity_root,
                timeout_seconds=spec.budget.evaluator_timeout_seconds,
                run_seed=run_seed,
            )
            evaluation = evaluated.evaluation
        else:
            replicates = []
            for replicate in range(1, replicate_count + 1):
                derived_seed = int.from_bytes(
                    hashlib.sha256(
                        f"v3-evaluator:{run_seed}:{replicate}".encode()
                    ).digest()[:8],
                    "big",
                )
                artifacts = evaluator_for(task).evaluate(
                    candidate_snapshot=candidate_snapshot,
                    opportunity_root=(
                        opportunity_root
                        / "evaluation-replicates"
                        / f"seed-{replicate:02d}"
                    ),
                    timeout_seconds=spec.budget.evaluator_timeout_seconds,
                    run_seed=derived_seed,
                )
                replicates.append((derived_seed, artifacts.evaluation))
            valid = all(item.valid for _, item in replicates)
            numeric_metrics: dict[str, float] = {}
            for name in task.public_feedback_metrics:
                values = [
                    item.metrics.get(name)
                    for _, item in replicates
                    if isinstance(item.metrics.get(name), int | float)
                    and not isinstance(item.metrics.get(name), bool)
                ]
                if len(values) == replicate_count:
                    numeric_metrics[name] = sum(float(value) for value in values) / len(
                        values
                    )
            fitness_values = [
                float(item.fitness)
                for _, item in replicates
                if item.valid and item.fitness is not None
            ]
            evaluation = Evaluation(
                valid=valid,
                fitness=(sum(fitness_values) / len(fitness_values) if valid else None),
                metrics={
                    **numeric_metrics,
                    "training_seed_replicates": [
                        {
                            "seed": seed,
                            "valid": item.valid,
                            "fitness": item.fitness,
                            "metrics": item.metrics,
                            "failure_kind": item.failure_kind,
                        }
                        for seed, item in replicates
                    ],
                },
                evaluator_seconds=sum(item.evaluator_seconds for _, item in replicates),
                evaluator_calls=1,
                failure_kind=None if valid else "replicate_evaluation_failure",
            )
    external_search = None
    if native_selection is not None:
        native_prompt = proposal.framework_metadata.get("native_prompt")
        external_search = stage_native_outcome(
            run_dir,
            controller=controller,
            task=task,
            framework=framework,
            vendor_root=repo_root / "architecture_discovery/vendor/openevolve",
            run_seed=run_seed,
            selection=native_selection,
            candidate_id=recorded_candidate_id,
            candidate_snapshot=candidate_snapshot,
            hypothesis=proposal.hypothesis,
            intended_edit=proposal.intended_edit,
            changes_summary=str(
                proposal.framework_metadata.get(
                    "changes_summary", proposal.intended_edit
                )
            ),
            evaluation=evaluation,
            prompt=(dict(native_prompt) if isinstance(native_prompt, dict) else None),
            response=(
                str(proposal.framework_metadata["llm_response"])
                if "llm_response" in proposal.framework_metadata
                else None
            ),
        )
    record = controller.complete(
        candidate_id=recorded_candidate_id,
        artifact_path=str(candidate_snapshot.relative_to(run_dir)),
        hypothesis=proposal.hypothesis,
        intended_edit=proposal.intended_edit,
        evaluation=evaluation,
        usage=proposal.codex.usage,
        prompt_hashes=_hashes(rendered),
        codex_service_tier=proposal.codex.service_tier,
        mechanism=proposal.mechanism,
        evidence=proposal.evidence,
        external_search=external_search,
    )
    if native_selection is not None:
        finalize_native_outcome(
            run_dir,
            opportunity=active.index,
            candidate_id=recorded_candidate_id,
        )
    if spec.protocol_version == "3.0":
        proposal_type = (
            "assumption_changing" if active.transition_active else "ordinary"
        )
        provenance = record_candidate_provenance(
            run_dir=run_dir,
            task=task,
            opportunity=active.index,
            parent_id=active.selected_parent_id,
            candidate_id=candidate_id,
            proposal_type=proposal_type,
            hypothesis=proposal.hypothesis,
            intended_edit=proposal.intended_edit,
            mechanism=proposal.mechanism,
            evidence=proposal.evidence,
        )
        if active.index in spec.transition_opportunities:
            write_manipulation_packet(
                campaign=run_dir.parent.parent,
                run_dir=run_dir,
                opportunity=active.index,
                candidate_id=candidate_id,
                proposal_type=proposal_type,
                provenance=provenance,
            )
        append_record = {
            "schema_version": "3.0",
            "event": "v3_proposal_provenance",
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": controller.state.run_id,
            "condition": controller.state.condition,
            "opportunity": active.index,
            "runtime_sha256": expected_runtime_hash,
            "prompt_bundle_sha256": run_manifest.get("campaign_prompt_bundle_sha256"),
            "conversation_boundary": (
                f"phased_session_span_{session_span}"
                if phased_session
                else "fresh_bounded_state_capsule"
            ),
            "provider_model_requested": spec.model.name,
            "reasoning_effort_requested": spec.model.reasoning_effort,
            "service_tier_observed": proposal.codex.service_tier,
        }
        from .state import append_jsonl

        append_jsonl(run_dir / "events.jsonl", append_record)
        developmental_options = dict(v3_options.get("developmental_reward", {}))
        if developmental_options:
            assessment = assess_developmental_value(
                run_dir=run_dir,
                task=task,
                record=record,
                provenance=provenance,
                config=developmental_options,
            )
            if assessment:
                record["developmental_assessment"] = assessment
    if workspace.exists():
        _make_tree_owner_writable(workspace)
        shutil.rmtree(workspace)
    (opportunity_root / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def run_one_opportunity(
    run_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
    allow_v3_prefix_leader: bool = False,
    agent_worker_lease: AgentWorkerLease | None = None,
) -> dict[str, object]:
    resolved = Path(run_dir).resolve()
    next_opportunity = SearchController.load(resolved, spec).state.next_opportunity
    lease = agent_worker_lease
    if lease is None:
        lease = acquire_agent_worker_slot(
            worker_id=f"{resolved}:{next_opportunity}",
            metadata={
                "campaign": str(resolved.parent.parent),
                "run_id": resolved.name,
                "opportunity": next_opportunity,
                "framework_id": str(framework.framework_id.value),
                "task_id": task.task_id,
            },
            cancel_path=resolved / "pause-request.json",
        )
    try:
        with _run_lock(resolved):
            return _run_one_opportunity_unlocked(
                resolved,
                spec=spec,
                task=task,
                framework=framework,
                repo_root=repo_root,
                python_bin=python_bin,
                codex_binary=codex_binary,
                codex_timeout_seconds=codex_timeout_seconds,
                allow_v3_prefix_leader=allow_v3_prefix_leader,
                agent_worker_lease=lease,
            )
    finally:
        release_agent_worker_slot(lease)


def recover_active_opportunity(
    run_dir: str | Path,
    *,
    spec: FactorialSpec,
    reason: str,
) -> dict[str, object]:
    """Close an interrupted active opportunity as a predeclared failure.

    Recovery never deletes or reuses its artifact directory and never retries
    the charged proposal. Any completed Codex usage is recovered from JSONL.
    """

    resolved = Path(run_dir).resolve()
    if not reason.strip():
        raise ValueError("recovery reason cannot be blank")
    with _run_lock(resolved):
        controller = SearchController.load(resolved, spec)
        active = controller.state.active
        if active is None:
            raise RuntimeError("run has no active opportunity to recover")
        opportunity_root = resolved / "opportunities" / f"{active.index:04d}"
        opportunity_root.mkdir(parents=True, exist_ok=True)
        events = opportunity_root / "codex" / f"proposal-{active.index}.jsonl"
        usage = usage_from_events(events)
        phased_session = False
        if spec.protocol_version == "3.0":
            try:
                phased_session = (
                    int(
                        dict(
                            load_runtime_options(resolved.parent.parent).get(
                                "conversation", {}
                            )
                        ).get("session_span_opportunities", 1)
                    )
                    > 1
                )
            except (OSError, ValueError):
                phased_session = False
        if spec.conversation_mode is ConversationMode.CONTINUOUS or phased_session:
            session_id = session_id_from_events(events)
            if session_id is not None:
                _register_conversation_session(
                    resolved,
                    session_id,
                    allow_multiple_for_run=phased_session,
                )
                controller.record_conversation_session(session_id)
        prompt_manifest = opportunity_root / "prompt_manifest.json"
        prompt_hashes = (
            json.loads(prompt_manifest.read_text(encoding="utf-8"))
            if prompt_manifest.is_file()
            else {}
        )
        identifier = hashlib.sha256(
            (
                f"recovered:{controller.state.run_id}:{active.index}:{reason.strip()}"
            ).encode()
        ).hexdigest()
        record = controller.complete(
            candidate_id=identifier,
            artifact_path=f"candidates/{active.selected_parent_id}",
            hypothesis="[interrupted before a complete proposal was recorded]",
            intended_edit="[not recoverable]",
            evaluation=Evaluation(
                valid=False,
                fitness=None,
                metrics={"recovery_reason": reason.strip()},
                evaluator_seconds=0.0,
                evaluator_calls=0,
                failure_kind="infrastructure_interruption",
            ),
            usage=usage,
            prompt_hashes=prompt_hashes,
        )
        (opportunity_root / "recovery.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record
