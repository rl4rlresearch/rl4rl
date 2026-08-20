"""Execute one resumable proposal opportunity through the shared controller."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .artifacts import (
    candidate_hash,
    make_read_only,
    materialize_candidate,
    protected_hash,
    snapshot_candidate,
)
from .codex_cli import CodexCli
from .evaluator import CommandEvaluator
from .frameworks import make_framework_adapter
from .prompts import PromptContext, PromptRenderer, VisibleCandidate
from .spec import FactorialSpec, FrameworkSpec, TaskSpec
from .state import Evaluation, SearchController


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hashes(rendered) -> dict[str, str]:
    return {
        "common_template_sha256": rendered.common_template_sha256,
        "search_state_sha256": rendered.search_state_sha256,
        "proposal_policy_sha256": rendered.proposal_policy_sha256,
        "prompt_sha256": rendered.prompt_sha256,
    }


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
    run_dir = Path(run_dir).resolve()
    controller = SearchController.load(run_dir, spec)
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
    visible_root = workspace / ".factorial-visible"
    visible_root.mkdir()
    visible_workspaces: list[Path] = []
    visible_candidates: list[VisibleCandidate] = []
    visible_records: list[dict[str, object]] = []
    for index, identifier in enumerate(active.visible_ids, start=1):
        candidate = controller.state.candidates[identifier]
        destination = visible_root / f"slot-{index}"
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
                artifact_path=f".factorial-visible/slot-{index}",
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
    prompt_context = PromptContext(
        condition=controller.condition,
        opportunity=active.index,
        selected_parent_id=active.selected_parent_id,
        visible_candidates=tuple(visible_candidates),
        remaining_proposals=int(remaining["proposals"]),
        remaining_evaluations=int(remaining["evaluations"]),
        remaining_tokens=int(remaining["tokens"]),
        remaining_evaluator_seconds=float(remaining["evaluator_seconds"]),
        no_search=controller.state.no_search,
    )
    renderer = PromptRenderer(repo_root / "experiments/c0c3_factorial/templates")
    rendered = renderer.render(spec, task, framework, prompt_context)
    (opportunity_root / "prompt.md").write_text(rendered.text, encoding="utf-8")
    (opportunity_root / "prompt_manifest.json").write_text(
        json.dumps(_hashes(rendered), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before_protected = protected_hash(workspace, task.editable_paths)
    parent_hash = candidate_hash(workspace, task.editable_paths)
    adapter = make_framework_adapter(
        framework, CodexCli(codex_binary), repo_root=repo_root
    )
    proposal = adapter.propose(
        rendered=rendered,
        workspace=workspace,
        model=spec.model,
        log_root=opportunity_root / "codex",
        call_id=f"proposal-{active.index}",
        timeout_seconds=codex_timeout_seconds,
        task=task,
        visible_workspaces=tuple(visible_workspaces),
        selected_parent_id=active.selected_parent_id,
        visible_records=tuple(visible_records),
    )
    protected_changed = (
        protected_hash(workspace, task.editable_paths) != before_protected
    )
    adapter_error = proposal.adapter_error
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
    if proposal.codex.returncode != 0 or adapter_error:
        evaluation = Evaluation(
            valid=False,
            fitness=None,
            metrics={
                "codex_returncode": proposal.codex.returncode,
                "adapter_error": adapter_error,
                "protected_changed": protected_changed,
            },
            evaluator_seconds=0.0,
            evaluator_calls=0,
            failure_kind=(
                "provider" if proposal.codex.returncode != 0 else "invalid_candidate"
            ),
        )
    else:
        evaluator = CommandEvaluator(
            task=task,
            support_source=support_source,
            repo_root=repo_root,
            python_bin=python_bin,
        )
        evaluated = evaluator.evaluate(
            candidate_snapshot=candidate_snapshot,
            opportunity_root=opportunity_root,
            timeout_seconds=spec.budget.evaluator_timeout_seconds,
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
    )
    shutil.rmtree(workspace, ignore_errors=True)
    (opportunity_root / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
