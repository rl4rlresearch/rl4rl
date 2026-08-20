"""Crash-safe search-state, retention, and accounting for all frameworks."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .spec import Condition, FactorialSpec


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class Candidate:
    candidate_id: str
    parent_ids: list[str]
    fitness: float
    metrics: dict[str, float | int | str | bool | None]
    artifact_path: str
    hypothesis: str
    intended_edit: str
    created_opportunity: int
    retained_order: int
    selected_count: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be blank")
        if not math.isfinite(self.fitness):
            raise ValueError("fitness must be finite")


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> None:
        for name in asdict(self):
            value = getattr(other, name)
            if value < 0:
                raise ValueError("token usage cannot be negative")
            setattr(self, name, getattr(self, name) + value)


@dataclass
class ActiveOpportunity:
    index: int
    visible_ids: list[str]
    selected_parent_id: str
    transition_active: bool
    started_at: str


@dataclass
class RunState:
    schema_version: str
    run_id: str
    condition: str
    protocol_hash: str
    no_search: bool
    status: str
    next_opportunity: int
    proposals_used: int
    evaluations_used: int
    evaluator_seconds_used: float
    usage: Usage
    incumbent_id: str
    portfolio_ids: list[str]
    candidates: dict[str, Candidate]
    active: ActiveOpportunity | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunState:
        payload = dict(value)
        payload["usage"] = Usage(**payload["usage"])
        payload["candidates"] = {
            key: Candidate(**candidate)
            for key, candidate in payload["candidates"].items()
        }
        if payload.get("active") is not None:
            payload["active"] = ActiveOpportunity(**payload["active"])
        return cls(**payload)


@dataclass(frozen=True)
class Evaluation:
    valid: bool
    fitness: float | None
    metrics: dict[str, float | int | str | bool | None]
    evaluator_seconds: float
    evaluator_calls: int = 1
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        if self.evaluator_calls not in {0, 1}:
            raise ValueError("a proposal may start at most one evaluator call")
        if self.evaluator_seconds < 0 or not math.isfinite(self.evaluator_seconds):
            raise ValueError("evaluator_seconds must be finite and nonnegative")
        if self.valid:
            if self.evaluator_calls != 1:
                raise ValueError("valid evaluations require one evaluator call")
            if self.fitness is None or not math.isfinite(self.fitness):
                raise ValueError("valid evaluations require finite fitness")
            if self.failure_kind is not None:
                raise ValueError("valid evaluations cannot have a failure kind")
        elif self.fitness is not None:
            raise ValueError("invalid evaluations cannot expose fitness")


class SearchController:
    """The one source of truth for C0-C3 state and budget transitions."""

    def __init__(self, run_dir: str | Path, spec: FactorialSpec, state: RunState):
        self.run_dir = Path(run_dir).resolve()
        self.spec = spec
        self.state = state
        if state.protocol_hash != spec.protocol_hash:
            raise ValueError("run state does not match the frozen protocol")
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self._validate()

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        spec: FactorialSpec,
        *,
        run_id: str,
        condition: Condition,
        seed_candidate: Candidate,
        no_search: bool = False,
    ) -> SearchController:
        destination = Path(run_dir).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        if seed_candidate.parent_ids:
            raise ValueError("seed candidate cannot have parents")
        state = RunState(
            schema_version="1.0",
            run_id=run_id,
            condition="N0" if no_search else condition.value,
            protocol_hash=spec.protocol_hash,
            no_search=no_search,
            status="ready",
            next_opportunity=1,
            proposals_used=0,
            evaluations_used=0,
            evaluator_seconds_used=0.0,
            usage=Usage(),
            incumbent_id=seed_candidate.candidate_id,
            portfolio_ids=[seed_candidate.candidate_id],
            candidates={seed_candidate.candidate_id: seed_candidate},
        )
        controller = cls(destination, spec, state)
        controller._write_state()
        append_jsonl(
            controller.events_path,
            {
                "schema_version": "1.0",
                "event": "run_created",
                "timestamp": utc_now(),
                "run_id": run_id,
                "condition": "N0" if no_search else condition.value,
                "no_search": no_search,
                "protocol_hash": spec.protocol_hash,
                "seed_candidate_id": seed_candidate.candidate_id,
            },
        )
        return controller

    @classmethod
    def load(cls, run_dir: str | Path, spec: FactorialSpec) -> SearchController:
        destination = Path(run_dir).resolve()
        state = RunState.from_dict(
            json.loads((destination / "state.json").read_text(encoding="utf-8"))
        )
        return cls(destination, spec, state)

    @property
    def condition(self) -> Condition:
        return Condition.C0 if self.state.no_search else Condition(self.state.condition)

    def _validate(self) -> None:
        state = self.state
        if state.schema_version != "1.0":
            raise ValueError("unsupported run-state version")
        if state.status not in {"ready", "running", "completed"}:
            raise ValueError("invalid run status")
        if not 1 <= state.next_opportunity <= self.spec.budget.proposals + 1:
            raise ValueError("next opportunity is outside the frozen budget")
        if state.proposals_used != state.next_opportunity - 1:
            raise ValueError("proposal opportunity accounting diverged")
        if not 0 <= state.evaluations_used <= state.proposals_used:
            raise ValueError("evaluation accounting exceeds proposal accounting")
        if state.incumbent_id not in state.candidates:
            raise ValueError("incumbent candidate is missing")
        if (
            not state.portfolio_ids
            or len(state.portfolio_ids) > self.spec.portfolio_capacity
        ):
            raise ValueError("portfolio violates frozen capacity")
        if len(set(state.portfolio_ids)) != len(state.portfolio_ids):
            raise ValueError("portfolio contains duplicates")
        if any(value not in state.candidates for value in state.portfolio_ids):
            raise ValueError("portfolio references unknown candidate")
        if not self.condition.has_portfolio and state.portfolio_ids != [
            state.incumbent_id
        ]:
            raise ValueError("single-incumbent condition exposed multiple candidates")
        # One in-flight call may overshoot the token ceiling. Completion is
        # allowed, but no new opportunity can begin.
        if (
            state.usage.total_tokens > self.spec.budget.max_total_tokens
            and state.active is not None
        ):
            raise ValueError("active call exists after token budget exhaustion")
        if state.active is not None:
            if state.active.index != state.next_opportunity:
                raise ValueError("active opportunity index is inconsistent")
            if state.active.selected_parent_id not in state.active.visible_ids:
                raise ValueError("selected parent must be visible")

    def _write_state(self) -> None:
        self._validate()
        self.state.revision += 1
        atomic_json(self.state_path, self.state.to_dict())

    def remaining(self) -> dict[str, int | float]:
        return {
            "proposals": self.spec.budget.proposals - self.state.proposals_used,
            "evaluations": (
                self.spec.budget.candidate_evaluations - self.state.evaluations_used
            ),
            "tokens": max(
                0, self.spec.budget.max_total_tokens - self.state.usage.total_tokens
            ),
            "evaluator_seconds": max(
                0.0,
                self.spec.budget.max_evaluator_seconds
                - self.state.evaluator_seconds_used,
            ),
        }

    def _selected_parent(self, visible: list[str]) -> str:
        if self.state.no_search:
            # Independent proposals always start at the frozen seed.
            return min(
                self.state.candidates.values(),
                key=lambda candidate: candidate.created_opportunity,
            ).candidate_id
        if not self.condition.has_portfolio:
            return self.state.incumbent_id
        return min(
            (self.state.candidates[identifier] for identifier in visible),
            key=lambda candidate: (
                candidate.selected_count,
                -candidate.fitness,
                candidate.retained_order,
                candidate.candidate_id,
            ),
        ).candidate_id

    def begin(self) -> ActiveOpportunity:
        if self.state.active is not None:
            raise RuntimeError("an opportunity is already active")
        remaining = self.remaining()
        if (
            remaining["proposals"] <= 0
            or remaining["evaluations"] <= 0
            or remaining["tokens"] <= 0
            or remaining["evaluator_seconds"] <= 0
        ):
            self.state.status = "completed"
            self._write_state()
            raise RuntimeError("the frozen run budget is exhausted")
        visible = (
            [self.state.incumbent_id]
            if not self.condition.has_portfolio or self.state.no_search
            else list(self.state.portfolio_ids)
        )
        parent_id = self._selected_parent(visible)
        if not self.state.no_search:
            self.state.candidates[parent_id].selected_count += 1
        active = ActiveOpportunity(
            index=self.state.next_opportunity,
            visible_ids=visible,
            selected_parent_id=parent_id,
            transition_active=(
                False
                if self.state.no_search
                else self.condition.transition_active(
                    self.state.next_opportunity,
                    self.spec.transition_opportunities,
                )
            ),
            started_at=utc_now(),
        )
        self.state.active = active
        self.state.status = "running"
        self._write_state()
        append_jsonl(
            self.events_path,
            {
                "schema_version": "1.0",
                "event": "proposal_started",
                "timestamp": active.started_at,
                "run_id": self.state.run_id,
                "condition": self.state.condition,
                "opportunity": active.index,
                "visible_candidate_ids": active.visible_ids,
                "selected_parent_ids": [active.selected_parent_id],
                "proposal_type": (
                    "independent_no_search"
                    if self.state.no_search
                    else (
                        "assumption_changing"
                        if active.transition_active
                        else "ordinary"
                    )
                ),
                "remaining_budget": remaining,
            },
        )
        return active

    def complete(
        self,
        *,
        candidate_id: str,
        artifact_path: str,
        hypothesis: str,
        intended_edit: str,
        evaluation: Evaluation,
        usage: Usage,
        prompt_hashes: dict[str, str],
    ) -> dict[str, Any]:
        active = self.state.active
        if active is None:
            raise RuntimeError("no opportunity is active")
        if candidate_id in self.state.candidates:
            raise ValueError("candidate ID has already been evaluated")
        self.state.usage.add(usage)
        self.state.proposals_used += 1
        self.state.evaluations_used += evaluation.evaluator_calls
        self.state.evaluator_seconds_used += evaluation.evaluator_seconds
        retained = False
        decision = "invalid"
        evicted_id: str | None = None
        if evaluation.valid:
            candidate = Candidate(
                candidate_id=candidate_id,
                parent_ids=[active.selected_parent_id],
                fitness=float(evaluation.fitness),
                metrics=evaluation.metrics,
                artifact_path=artifact_path,
                hypothesis=hypothesis,
                intended_edit=intended_edit,
                created_opportunity=active.index,
                retained_order=active.index,
            )
            self.state.candidates[candidate_id] = candidate
            parent = self.state.candidates[active.selected_parent_id]
            if self.state.no_search:
                decision = "independent_not_retained"
            elif not self.condition.has_portfolio:
                if candidate.fitness > parent.fitness:
                    retained = True
                    evicted_id = parent.candidate_id
                    self.state.incumbent_id = candidate_id
                    self.state.portfolio_ids = [candidate_id]
                    decision = "strict_incumbent_improvement"
                else:
                    decision = "not_strictly_better_than_incumbent"
            elif len(self.state.portfolio_ids) < self.spec.portfolio_capacity:
                retained = True
                self.state.portfolio_ids.append(candidate_id)
                decision = "filled_open_portfolio_slot"
            elif candidate.fitness > parent.fitness:
                retained = True
                position = self.state.portfolio_ids.index(parent.candidate_id)
                self.state.portfolio_ids[position] = candidate_id
                evicted_id = parent.candidate_id
                decision = "replaced_selected_parent_on_strict_improvement"
            else:
                decision = "not_strictly_better_than_selected_parent"
            if self.condition.has_portfolio and not self.state.no_search:
                self.state.incumbent_id = max(
                    self.state.portfolio_ids,
                    key=lambda identifier: (
                        self.state.candidates[identifier].fitness,
                        -self.state.candidates[identifier].retained_order,
                        identifier,
                    ),
                )
        remaining_before_advance = self.remaining()
        record = {
            "schema_version": "1.0",
            "event": "proposal_completed",
            "timestamp": utc_now(),
            "run_id": self.state.run_id,
            "condition": self.state.condition,
            "opportunity": active.index,
            "visible_candidate_ids": active.visible_ids,
            "selected_parent_ids": [active.selected_parent_id],
            "proposal_type": (
                "independent_no_search"
                if self.state.no_search
                else ("assumption_changing" if active.transition_active else "ordinary")
            ),
            "hypothesis": hypothesis,
            "intended_edit": intended_edit,
            "candidate_id": candidate_id,
            "parent_ids": [active.selected_parent_id],
            "artifact_path": artifact_path,
            "evaluation": asdict(evaluation),
            "retained": retained,
            "retention_decision": decision,
            "evicted_candidate_id": evicted_id,
            "portfolio_after": list(self.state.portfolio_ids),
            "incumbent_after": self.state.incumbent_id,
            "usage_increment": asdict(usage) | {"total_tokens": usage.total_tokens},
            "usage_cumulative": asdict(self.state.usage)
            | {"total_tokens": self.state.usage.total_tokens},
            "evaluator_calls_increment": evaluation.evaluator_calls,
            "evaluator_calls_cumulative": self.state.evaluations_used,
            "proposals_cumulative": self.state.proposals_used,
            "evaluator_seconds_increment": evaluation.evaluator_seconds,
            "evaluator_seconds_cumulative": self.state.evaluator_seconds_used,
            "remaining_budget": remaining_before_advance,
            "prompt_hashes": dict(sorted(prompt_hashes.items())),
        }
        append_jsonl(self.events_path, record)
        self.state.active = None
        self.state.next_opportunity += 1
        exhausted = (
            self.state.next_opportunity > self.spec.budget.proposals
            or self.state.usage.total_tokens >= self.spec.budget.max_total_tokens
            or self.state.evaluator_seconds_used
            >= self.spec.budget.max_evaluator_seconds
        )
        if exhausted:
            self.state.status = "completed"
        self._write_state()
        return record
