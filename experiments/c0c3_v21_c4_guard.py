"""C4 periodic-refresh compatibility for hash-pinned v2.1 runtimes.

The original v2.1 campaign runtimes know only C0-C3 and must remain byte-for-
byte frozen.  The operator-authorized C4 amendment is therefore installed by
the external supervisor.  It gives C4 C0's single-incumbent ordinary proposal
policy and performs an incumbent-preserving, history-free search restart after
each ten completed proposals.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

C4 = "C4"
MEMORY_STATE = Path("subject-memory.json")
DEFAULT_INTERVAL = 10


class _C4Condition:
    value = C4
    has_portfolio = False

    def transition_active(self, opportunity: int, schedule: Any) -> bool:
        del opportunity, schedule
        return False


C4_CONDITION = _C4Condition()
_ACTIVE_C4_RUN: ContextVar[Path | None] = ContextVar(
    "rl4rl_active_c4_run", default=None
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_c4_run(run_dir: Path) -> bool:
    manifest = _read_object(run_dir / "manifest.json")
    assignment = manifest.get("assignment")
    return isinstance(assignment, dict) and assignment.get("condition") == C4


def _memory_state(run_dir: Path, *, base_seed: int) -> dict[str, Any]:
    current = _read_object(run_dir / MEMORY_STATE)
    if current:
        return current
    return {
        "schema_version": "1.0",
        "history_start_opportunity": 1,
        "search_seed": base_seed,
        "phase": 1,
        "interval_proposals": DEFAULT_INTERVAL,
    }


def _search_seed(base_seed: int, opportunity: int) -> int:
    payload = (
        f"greedy-openevolve-v2.1-c4-refresh\0{base_seed}\0{opportunity}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _refresh_if_due(
    run_dir: Path,
    *,
    spec: Any,
    state_module: ModuleType,
) -> None:
    if not _is_c4_run(run_dir):
        return
    manifest = _read_object(run_dir / "manifest.json")
    policy = manifest.get("periodic_full_refresh")
    if not isinstance(policy, dict):
        raise ValueError("C4 run lacks its periodic full-refresh policy")
    interval = int(policy.get("interval_proposals", DEFAULT_INTERVAL))
    if interval < 1:
        raise ValueError("C4 refresh interval must be positive")
    controller = state_module.SearchController.load(run_dir, spec)
    opportunity = int(controller.state.next_opportunity)
    if opportunity <= 1 or (opportunity - 1) % interval != 0:
        return
    memory = _memory_state(
        run_dir,
        base_seed=int(manifest["assignment"]["run_seed"]),
    )
    if int(memory.get("history_start_opportunity", 1)) == opportunity:
        return
    if controller.state.active is not None:
        raise RuntimeError("cannot refresh an active C4 opportunity")

    incumbent = controller.state.candidates[controller.state.incumbent_id]
    prior_candidate_ids = sorted(controller.state.candidates)
    prior_portfolio_ids = list(controller.state.portfolio_ids)
    prior_session_id = controller.state.conversation_session_id
    seed = _search_seed(int(manifest["assignment"]["run_seed"]), opportunity)
    refreshed = state_module.Candidate(
        candidate_id=incumbent.candidate_id,
        parent_ids=[],
        fitness=incumbent.fitness,
        metrics=dict(incumbent.metrics),
        artifact_path=incumbent.artifact_path,
        hypothesis="starting design",
        intended_edit="none",
        created_opportunity=opportunity - 1,
        retained_order=0,
        selected_count=0,
    )
    controller.state.candidates = {refreshed.candidate_id: refreshed}
    controller.state.portfolio_ids = [refreshed.candidate_id]
    controller.state.conversation_session_id = None
    controller._write_state()
    state_module.append_jsonl(
        controller.events_path,
        {
            "schema_version": "1.0",
            "event": "search_epoch_refreshed_from_incumbent",
            "timestamp": state_module.utc_now(),
            "run_id": controller.state.run_id,
            "opportunity": opportunity,
            "incumbent_id": refreshed.candidate_id,
            "prior_candidate_ids": prior_candidate_ids,
            "prior_portfolio_ids": prior_portfolio_ids,
            "prior_conversation_session_id": prior_session_id,
            "search_seed": seed,
            "reason": "configured ten-proposal C4 full search refresh",
        },
    )
    archive_path = run_dir / "developmental-archive.json"
    if archive_path.is_file():
        state_module.append_jsonl(
            run_dir / "developmental-archive-resets.jsonl",
            {
                "schema_version": "1.0",
                "event": "developmental_archive_reset",
                "timestamp": state_module.utc_now(),
                "opportunity": opportunity,
                "archive": _read_object(archive_path),
            },
        )
        state_module.atomic_json(
            archive_path,
            {
                "schema_version": "1.0",
                "items": [],
                "history_start_opportunity": opportunity,
            },
        )
    state_module.atomic_json(
        run_dir / MEMORY_STATE,
        {
            "schema_version": "1.0",
            "policy": "refresh_incumbent_every_ten_proposals",
            "history_start_opportunity": opportunity,
            "search_seed": seed,
            "phase": 1 + (opportunity - 1) // interval,
            "interval_proposals": interval,
            "incumbent_id": refreshed.candidate_id,
            "updated_at": state_module.utc_now(),
        },
    )


def _minimum_visible_opportunity(run_dir: Path) -> int:
    if not _is_c4_run(run_dir):
        return 1
    return int(_read_object(run_dir / MEMORY_STATE).get("history_start_opportunity", 1))


def _epoch_accounting(run_dir: Path, *, minimum: int) -> dict[str, float]:
    totals = {"evaluations": 0.0, "tokens": 0.0, "evaluator_seconds": 0.0}
    events = run_dir / "events.jsonl"
    if not events.is_file():
        return totals
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "proposal_completed":
            continue
        if int(record.get("opportunity", 0)) < minimum:
            continue
        totals["evaluations"] += float(record.get("evaluator_calls_increment", 0))
        totals["evaluator_seconds"] += float(
            record.get("evaluator_seconds_increment", 0.0)
        )
        usage = record.get("usage_increment")
        if isinstance(usage, dict):
            totals["tokens"] += float(usage.get("total_tokens", 0))
    return totals


def _fresh_prompt_context(
    context: Any,
    *,
    spec: Any,
    run_dir: Path,
) -> Any:
    """Return the subject-visible context for the current ten-proposal epoch.

    The physical counters in ``state.json`` remain cumulative for enforcement
    and audit.  Only the context rendered for the research subject is rebased.
    """

    minimum = _minimum_visible_opportunity(run_dir)
    local_opportunity = int(context.opportunity) - minimum + 1
    epoch = _epoch_accounting(run_dir, minimum=minimum)
    return replace(
        context,
        opportunity=local_opportunity,
        remaining_proposals=max(
            0, int(spec.budget.proposals) - local_opportunity + 1
        ),
        remaining_evaluations=max(
            0,
            int(spec.budget.candidate_evaluations) - int(epoch["evaluations"]),
        ),
        remaining_tokens=max(
            0,
            int(spec.budget.max_total_tokens) - int(epoch["tokens"]),
        ),
        remaining_evaluator_seconds=max(
            0.0,
            float(spec.budget.max_evaluator_seconds)
            - epoch["evaluator_seconds"],
        ),
    )


def _make_tree_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except FileNotFoundError:
            continue


def _physical_active_opportunity(run_dir: Path) -> int:
    state = _read_object(run_dir / "state.json")
    active = state.get("active")
    if not isinstance(active, dict):
        raise RuntimeError("C4 Codex call has no active controller opportunity")
    index = active.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise RuntimeError("C4 active controller opportunity is invalid")
    return index


def _mechanism_ledger(
    run_dir: Path, *, limit: int = 24, minimum_opportunity: int = 1
) -> str:
    minimum = max(_minimum_visible_opportunity(run_dir), minimum_opportunity)
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
        if int(record.get("opportunity", 0)) < minimum:
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
        metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
        if valid and isinstance(metrics, dict) and isinstance(
            metrics.get("parameters"), int | float
        ):
            parameters = int(metrics["parameters"])
            prior = summary.get("best_parameters")
            summary["best_parameters"] = (
                parameters if not isinstance(prior, int) else min(prior, parameters)
            )
    if not grouped:
        return "No earlier mechanism result is available."
    return "\n".join(
        f"- {grouped[key]['label']}: attempts={grouped[key]['attempts']}; "
        f"last={grouped[key]['last_result']}; "
        f"best_qualified_parameters={grouped[key].get('best_parameters', 'none')}"
        for key in order[-limit:]
    )


def install_v21_c4_guard(
    runner_module: ModuleType,
    state_module: ModuleType,
) -> None:
    """Install C4 support on a frozen v2.1 controller, idempotently."""

    if getattr(runner_module, "_rl4rl_v21_c4_guard_installed", False):
        return
    controller_class = state_module.SearchController
    original_condition = controller_class.condition.fget
    if original_condition is None:
        raise RuntimeError("frozen SearchController condition property is invalid")

    def condition(self: Any) -> Any:
        if self.state.condition == C4:
            return C4_CONDITION
        return original_condition(self)

    controller_class.condition = property(condition)

    original_recent_outcomes = runner_module._recent_outcomes

    def recent_outcomes(
        run_dir: Path, *, limit: int = 12, minimum_opportunity: int = 1
    ) -> tuple[Any, ...]:
        values = original_recent_outcomes(
            run_dir,
            limit=1_000_000,
            minimum_opportunity=minimum_opportunity,
        )
        minimum = max(
            _minimum_visible_opportunity(run_dir), minimum_opportunity
        )
        return tuple(
            value for value in values if int(value.opportunity) >= minimum
        )[-limit:]

    original_mechanism_ledger = runner_module._mechanism_ledger

    def mechanism_ledger(
        run_dir: Path, *, limit: int = 24, minimum_opportunity: int = 1
    ) -> str:
        if not _is_c4_run(run_dir):
            return original_mechanism_ledger(
                run_dir,
                limit=limit,
                minimum_opportunity=minimum_opportunity,
            )
        return _mechanism_ledger(
            run_dir,
            limit=limit,
            minimum_opportunity=minimum_opportunity,
        )

    renderer_class = runner_module.PromptRenderer
    original_render = renderer_class.render

    def render_with_fresh_accounting(
        self: Any,
        spec: Any,
        task: Any,
        framework: Any,
        context: Any,
    ) -> Any:
        run_dir = _ACTIVE_C4_RUN.get()
        if run_dir is not None:
            context = _fresh_prompt_context(context, spec=spec, run_dir=run_dir)
        return original_render(self, spec, task, framework, context)

    original_codex_run = runner_module.CodexCli.run

    def codex_run_with_opaque_workspace(
        self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        run_dir = _ACTIVE_C4_RUN.get()
        if run_dir is None:
            return original_codex_run(self, *args, **kwargs)
        workspace = Path(kwargs["workspace"]).resolve()
        minimum = _minimum_visible_opportunity(run_dir)
        physical_opportunity = _physical_active_opportunity(run_dir)
        local_opportunity = physical_opportunity - minimum + 1
        manifest = _read_object(run_dir / "manifest.json")
        base_seed = int(manifest["assignment"]["run_seed"])
        memory = _memory_state(run_dir, base_seed=base_seed)
        with tempfile.TemporaryDirectory(
            prefix="transformer-design-cycle-"
        ) as temporary:
            opaque = Path(temporary) / "workspace"
            shutil.copytree(workspace, opaque)
            rewritten = dict(kwargs)
            rewritten["workspace"] = opaque
            rewritten["call_id"] = f"proposal-{local_opportunity}"
            rewritten["run_seed"] = int(memory["search_seed"])
            try:
                return original_codex_run(self, *args, **rewritten)
            finally:
                _make_tree_writable(workspace)
                shutil.rmtree(workspace)
                shutil.copytree(opaque, workspace)

    original_unlocked = runner_module._run_one_opportunity_unlocked

    def run_one_opportunity_unlocked(
        run_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, object]:
        resolved = Path(run_dir).resolve()
        spec = kwargs.get("spec")
        if spec is None:
            raise TypeError("C4 compatibility runner requires keyword spec")
        _refresh_if_due(resolved, spec=spec, state_module=state_module)
        active_token = _ACTIVE_C4_RUN.set(resolved if _is_c4_run(resolved) else None)
        try:
            return original_unlocked(resolved, *args, **kwargs)
        finally:
            _ACTIVE_C4_RUN.reset(active_token)

    runner_module._recent_outcomes = recent_outcomes
    runner_module._mechanism_ledger = mechanism_ledger
    renderer_class.render = render_with_fresh_accounting
    runner_module.CodexCli.run = codex_run_with_opaque_workspace
    runner_module._run_one_opportunity_unlocked = run_one_opportunity_unlocked
    runner_module._rl4rl_v21_c4_guard_installed = True
