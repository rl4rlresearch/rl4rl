"""Execute one resumable proposal opportunity through the shared controller."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import (
    candidate_hash,
    make_read_only,
    materialize_candidate,
    protected_hash,
    scientific_runtime_hash,
    snapshot_candidate,
)
from .codex_cli import CodexCli, session_id_from_events, usage_from_events
from .evaluator import make_command_evaluator
from .frameworks import make_framework_adapter
from .neutral_task import (
    PAIR_TOKEN_TASK_ADAPTER_V2,
    SUBJECT_NEUTRAL_PROMPT_PROFILES,
)
from .prompts import (
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


def _register_conversation_session(run_dir: Path, session_id: str) -> None:
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
            if owned_ids and session_id not in owned_ids:
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
    }


def _opaque_run_id(run_dir: Path) -> str:
    return (
        hashlib.sha256(str(run_dir).encode()).hexdigest().translate(
            str.maketrans({"c": "w", "0": "q", "1": "r", "2": "s", "3": "t"})
        )[:24]
    )


def _refresh_continuous_workspace(
    *,
    run_dir: Path,
    support_source: Path,
    selected_snapshot: Path,
    editable_paths: tuple[str, ...],
    neutral_subject: bool = False,
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
    identity = hashlib.sha256(f"workspace:{run_dir}".encode()).hexdigest()
    (workspace / ".workspace-identity").write_text(identity + "\n", encoding="utf-8")
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("continuous Codex workspace contains a symlink")
    return workspace


def _make_tree_owner_writable(root: Path) -> None:
    """Allow safe cleanup of controller-created read-only reference trees."""

    paths = [root, *root.rglob("*")]
    for path in reversed(paths):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _recent_outcomes(run_dir: Path, *, limit: int = 12) -> tuple[VisibleOutcome, ...]:
    events = run_dir / "events.jsonl"
    if not events.is_file():
        return ()
    outcomes: list[VisibleOutcome] = []
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "proposal_completed":
            continue
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        metrics = evaluation.get("metrics")
        outcomes.append(
            VisibleOutcome(
                opportunity=int(record["opportunity"]),
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
            )
        )
    return tuple(outcomes[-limit:])


def _mechanism_ledger(run_dir: Path, *, limit: int = 24) -> str:
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
) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    if (
        spec.conversation_mode is ConversationMode.CONTINUOUS
        and framework.framework_id is not FrameworkKind.AUTORESEARCH
    ):
        raise ValueError(
            "continuous Codex sessions currently support Autoresearch only"
        )
    controller = SearchController.load(run_dir, spec)
    run_manifest = _read_json(run_dir / "manifest.json")
    expected_runtime_hash = scientific_runtime_hash(
        repo_root, task=task, framework=framework
    )
    if run_manifest.get("scientific_runtime_hash") != expected_runtime_hash:
        raise ValueError(
            "scientific runtime changed after campaign creation; create a new campaign"
        )
    assignment = run_manifest.get("assignment")
    if not isinstance(assignment, dict) or isinstance(
        assignment.get("run_seed"), bool
    ) or not isinstance(assignment.get("run_seed"), int):
        raise ValueError("run manifest lacks an integer assignment run_seed")
    run_seed = int(assignment["run_seed"])
    if controller.state.active is not None:
        raise RuntimeError(
            "run has an active opportunity; inspect its logs before explicit recovery"
        )
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
                "metrics": candidate.metrics,
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
        recent_outcomes=(
            ()
            if controller.state.no_search or not neutral_subject
            else _recent_outcomes(run_dir)
        ),
        mechanism_ledger=(
            _mechanism_ledger(run_dir)
            if framework.adapter == "controlled_openevolve_prompt_diff_v2"
            else "No earlier mechanism result is available."
        ),
    )
    renderer = PromptRenderer(repo_root / "experiments/c0c3_factorial/templates")
    rendered = renderer.render(spec, task, framework, prompt_context)
    (opportunity_root / "prompt.md").write_text(rendered.text, encoding="utf-8")
    (opportunity_root / "prompt_manifest.json").write_text(
        json.dumps(_hashes(rendered), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if show_token_continuation_notice:
        controller.record_token_budget_continuation_notice()
    continuous = spec.conversation_mode is ConversationMode.CONTINUOUS
    codex_workspace = (
        _refresh_continuous_workspace(
            run_dir=run_dir,
            support_source=support_source,
            selected_snapshot=selected_snapshot,
            editable_paths=task.editable_paths,
            neutral_subject=neutral_subject,
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
        framework, CodexCli(codex_binary), repo_root=repo_root
    )
    requested_session_id = (
        controller.state.conversation_session_id if continuous else None
    )
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
        run_seed=run_seed,
        resume_session_id=requested_session_id,
        persist_session=continuous,
        neutral_subject=neutral_subject,
    )
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
                    run_dir, proposal.codex.session_id
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
        and task.adapter == PAIR_TOKEN_TASK_ADAPTER_V2
    ):
        preflight_error = preflight_candidate_source(workspace)
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
        max_parallel_evaluators = EVALUATOR_CONCURRENCY_BY_PROTOCOL.get(
            spec.protocol_version
        )
        evaluator = make_command_evaluator(
            task=task,
            support_source=support_source,
            repo_root=repo_root,
            python_bin=python_bin,
            slot_root=(
                run_dir.parent.parent / ".evaluator-slots"
                if max_parallel_evaluators is not None
                else None
            ),
            max_parallel_evaluators=max_parallel_evaluators,
        )
        evaluated = evaluator.evaluate(
            candidate_snapshot=candidate_snapshot,
            opportunity_root=opportunity_root,
            timeout_seconds=spec.budget.evaluator_timeout_seconds,
            run_seed=run_seed,
        )
        evaluation = evaluated.evaluation
    record = controller.complete(
        candidate_id=recorded_candidate_id,
        artifact_path=str(candidate_snapshot.relative_to(run_dir)),
        hypothesis=proposal.hypothesis,
        intended_edit=proposal.intended_edit,
        evaluation=evaluation,
        usage=proposal.codex.usage,
        prompt_hashes=_hashes(rendered),
        mechanism=proposal.mechanism,
        evidence=proposal.evidence,
    )
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
) -> dict[str, object]:
    resolved = Path(run_dir).resolve()
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
        )


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
        if spec.conversation_mode is ConversationMode.CONTINUOUS:
            session_id = session_id_from_events(events)
            if session_id is not None:
                _register_conversation_session(resolved, session_id)
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
