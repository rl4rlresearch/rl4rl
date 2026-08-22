"""Frozen, portable contracts for the C0-C3 factorial experiment."""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class SearchState(StrEnum):
    SINGLE = "single_incumbent"
    PORTFOLIO = "portfolio_memory"


class ProposalPolicy(StrEnum):
    ORDINARY = "ordinary"
    SCHEDULED = "scheduled_assumption_changing"


class ConversationMode(StrEnum):
    """Whether each controller run uses fresh or resumed Codex context."""

    EPHEMERAL = "ephemeral_per_opportunity_v1"
    CONTINUOUS = "continuous_session_per_run_v1"


class Condition(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"

    @property
    def search_state(self) -> SearchState:
        return {
            Condition.C0: SearchState.SINGLE,
            Condition.C1: SearchState.SINGLE,
            Condition.C2: SearchState.PORTFOLIO,
            Condition.C3: SearchState.PORTFOLIO,
        }[self]

    @property
    def proposal_policy(self) -> ProposalPolicy:
        return {
            Condition.C0: ProposalPolicy.ORDINARY,
            Condition.C1: ProposalPolicy.SCHEDULED,
            Condition.C2: ProposalPolicy.ORDINARY,
            Condition.C3: ProposalPolicy.SCHEDULED,
        }[self]

    @property
    def has_portfolio(self) -> bool:
        return self.search_state is SearchState.PORTFOLIO

    def transition_active(self, opportunity: int, schedule: Iterable[int]) -> bool:
        return (
            self.proposal_policy is ProposalPolicy.SCHEDULED
            and opportunity in frozenset(schedule)
        )


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class FrameworkKind(StrEnum):
    AUTORESEARCH = "karpathy_autoresearch"
    OPENEVOLVE = "openevolve"


class ExecutionBackend(StrEnum):
    LOCAL = "local"
    MODAL = "modal"


PORTFOLIO_RETENTION_RULE = (
    "fill_open_slots_then_replace_selected_lineage_on_strict_improvement_v1"
)
PARENT_SELECTION_RULE = (
    "fill_from_seed_then_least_selected_lineage_then_best_then_oldest_then_id_v1"
)
SINGLE_RETENTION_RULE = "strict_incumbent_improvement_v1"
FAILURE_RULE = "consume_opportunity_and_evaluation_if_started;never_retain_v1"
SERIAL_EXECUTION_RULE = "blocked_round_robin_one_opportunity_v1"
PARALLEL_EXECUTION_RULE = "blocked_parallel_condition_rounds_v1"
STAGED_PARALLEL_EXECUTION_RULE = "staged_parallel_block_trajectories_v1"
STAGED_INDEPENDENT_EXECUTION_RULE = "staged_independent_parallel_trajectories_v1"
STAGED_INDIVIDUAL_EXECUTION_RULE = "staged_individually_controlled_trajectories_v1"
STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE = (
    "staged_confined_individually_controlled_trajectories_v1"
)
STAGED_EXECUTION_RULES = frozenset(
    {
        STAGED_PARALLEL_EXECUTION_RULE,
        STAGED_INDEPENDENT_EXECUTION_RULE,
        STAGED_INDIVIDUAL_EXECUTION_RULE,
        STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
    }
)
INDIVIDUAL_EXECUTION_RULES = frozenset(
    {
        STAGED_INDIVIDUAL_EXECUTION_RULE,
        STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
    }
)
# Protocol 1.6 permits every Codex trajectory to think concurrently while
# bounding local trainers. This avoids timeout artifacts from twelve training
# jobs oversubscribing one laptop and is identical in every factorial cell.
EVALUATOR_CONCURRENCY_BY_PROTOCOL = {"1.6": 3}
# Backward-compatible name for the original paper-v1 execution rule.
EXECUTION_RULE = SERIAL_EXECUTION_RULE
PROTOCOL_EXECUTION_RULES = {
    "1.0": SERIAL_EXECUTION_RULE,
    "1.1": PARALLEL_EXECUTION_RULE,
    "1.2": PARALLEL_EXECUTION_RULE,
    "1.3": STAGED_PARALLEL_EXECUTION_RULE,
    "1.4": STAGED_INDEPENDENT_EXECUTION_RULE,
    "1.5": STAGED_INDIVIDUAL_EXECUTION_RULE,
    "1.6": STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{name} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelSpec:
    name: str
    reasoning_effort: str
    sandbox: str = "workspace-write"
    approval_policy: str = "never"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name cannot be blank")
        if self.reasoning_effort not in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported reasoning effort")
        if self.sandbox not in {"read-only", "workspace-write"}:
            raise ValueError("scientific runs permit read-only or workspace-write only")
        if self.approval_policy != "never":
            raise ValueError("non-interactive runs require approval_policy='never'")


@dataclass(frozen=True)
class BudgetSpec:
    proposals: int
    candidate_evaluations: int
    max_total_tokens: int
    max_evaluator_seconds: float
    evaluator_timeout_seconds: int

    def __post_init__(self) -> None:
        _positive_int(self.proposals, "proposals")
        _positive_int(self.candidate_evaluations, "candidate_evaluations")
        _positive_int(self.max_total_tokens, "max_total_tokens")
        _positive_int(self.evaluator_timeout_seconds, "evaluator_timeout_seconds")
        if self.candidate_evaluations != self.proposals:
            raise ValueError("one candidate evaluation must be budgeted per proposal")
        if self.max_evaluator_seconds <= 0:
            raise ValueError("max_evaluator_seconds must be positive")


@dataclass(frozen=True)
class FactorialSpec:
    protocol_version: str
    study_id: str
    study_seed: int
    blocks: int
    portfolio_capacity: int
    transition_opportunities: tuple[int, ...]
    model: ModelSpec
    budget: BudgetSpec
    conversation_mode: ConversationMode = ConversationMode.EPHEMERAL
    retention_rule: str = PORTFOLIO_RETENTION_RULE
    parent_selection_rule: str = PARENT_SELECTION_RULE
    single_retention_rule: str = SINGLE_RETENTION_RULE
    failure_rule: str = FAILURE_RULE
    execution_rule: str = EXECUTION_RULE

    def __post_init__(self) -> None:
        if self.protocol_version not in PROTOCOL_EXECUTION_RULES:
            raise ValueError("unsupported protocol version")
        if not self.study_id or any(character.isspace() for character in self.study_id):
            raise ValueError("study_id must be non-empty and contain no whitespace")
        if isinstance(self.study_seed, bool) or not isinstance(self.study_seed, int):
            raise ValueError("study_seed must be an integer")
        _positive_int(self.blocks, "blocks")
        if self.portfolio_capacity < 2:
            raise ValueError("portfolio_capacity must be at least two")
        if self.retention_rule != PORTFOLIO_RETENTION_RULE:
            raise ValueError("unknown portfolio retention rule")
        if self.parent_selection_rule != PARENT_SELECTION_RULE:
            raise ValueError("unknown parent selection rule")
        if self.single_retention_rule != SINGLE_RETENTION_RULE:
            raise ValueError("unknown single-incumbent retention rule")
        if self.failure_rule != FAILURE_RULE:
            raise ValueError("unknown failure rule")
        expected_execution_rule = PROTOCOL_EXECUTION_RULES[self.protocol_version]
        if self.execution_rule != expected_execution_rule:
            raise ValueError(
                f"protocol {self.protocol_version} requires execution_rule="
                f"{expected_execution_rule!r}"
            )
        if (
            self.protocol_version in {"1.2", "1.3", "1.4", "1.5", "1.6"}
            and self.conversation_mode is not ConversationMode.CONTINUOUS
        ):
            raise ValueError(
                f"protocol {self.protocol_version} requires "
                "continuous_session_per_run_v1"
            )
        if (
            self.protocol_version not in {"1.2", "1.3", "1.4", "1.5", "1.6"}
            and self.conversation_mode is not ConversationMode.EPHEMERAL
        ):
            raise ValueError(
                "continuous Codex sessions require a separately versioned protocol"
            )
        schedule = self.transition_opportunities
        if tuple(sorted(set(schedule))) != schedule:
            raise ValueError("transition schedule must be sorted and unique")
        if not schedule:
            raise ValueError("transition schedule cannot be empty")
        if any(value < 1 or value > self.budget.proposals for value in schedule):
            raise ValueError("transition checkpoint outside proposal budget")

    @property
    def protocol_hash(self) -> str:
        return sha256_json(asdict(self))

    @classmethod
    def from_toml(cls, path: str | Path) -> FactorialSpec:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        payload.setdefault("conversation_mode", ConversationMode.EPHEMERAL.value)
        _strict_keys(
            payload,
            {
                "protocol_version",
                "study_id",
                "study_seed",
                "blocks",
                "portfolio_capacity",
                "transition_opportunities",
                "conversation_mode",
                "retention_rule",
                "parent_selection_rule",
                "single_retention_rule",
                "failure_rule",
                "execution_rule",
                "model",
                "budget",
            },
            "factorial protocol",
        )
        model = payload.pop("model")
        budget = payload.pop("budget")
        transition_opportunities = tuple(payload.pop("transition_opportunities"))
        conversation_mode = ConversationMode(payload.pop("conversation_mode"))
        _strict_keys(
            model,
            {"name", "reasoning_effort", "sandbox", "approval_policy"},
            "model",
        )
        _strict_keys(
            budget,
            {
                "proposals",
                "candidate_evaluations",
                "max_total_tokens",
                "max_evaluator_seconds",
                "evaluator_timeout_seconds",
            },
            "budget",
        )
        return cls(
            **payload,
            transition_opportunities=transition_opportunities,
            conversation_mode=conversation_mode,
            model=ModelSpec(**model),
            budget=BudgetSpec(**budget),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    display_name: str
    adapter: str
    seed_source: str
    editable_paths: tuple[str, ...]
    evaluator_command: tuple[str, ...]
    objective_metric: str
    objective_direction: ObjectiveDirection
    qualification_metric: str | None
    qualification_minimum: float | None
    public_feedback_metrics: tuple[str, ...]
    metric_patterns: dict[str, str]
    final_holdout_command: tuple[str, ...]
    preferred_backend: ExecutionBackend

    def __post_init__(self) -> None:
        if not self.task_id or any(character.isspace() for character in self.task_id):
            raise ValueError("task_id must be portable and contain no whitespace")
        if not self.editable_paths:
            raise ValueError("task must expose at least one editable path")
        for value in self.editable_paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("editable paths must be safe relative paths")
        if not self.evaluator_command or not self.final_holdout_command:
            raise ValueError("task requires search and final-holdout commands")
        if (self.qualification_metric is None) != (self.qualification_minimum is None):
            raise ValueError("qualification metric and threshold must be set together")
        if self.objective_metric not in self.public_feedback_metrics:
            raise ValueError("objective metric must be visible Layer A feedback")
        if self.qualification_metric is not None and (
            self.qualification_metric not in self.public_feedback_metrics
        ):
            raise ValueError("qualification metric must be visible Layer A feedback")

    @classmethod
    def from_toml(cls, path: str | Path) -> TaskSpec:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        payload.setdefault("qualification_metric", None)
        payload.setdefault("qualification_minimum", None)
        editable_paths = tuple(payload.pop("editable_paths"))
        evaluator_command = tuple(payload.pop("evaluator_command"))
        objective_direction = ObjectiveDirection(payload.pop("objective_direction"))
        public_feedback_metrics = tuple(payload.pop("public_feedback_metrics"))
        metric_patterns = dict(payload.pop("metric_patterns"))
        final_holdout_command = tuple(payload.pop("final_holdout_command"))
        preferred_backend = ExecutionBackend(payload.pop("preferred_backend"))
        return cls(
            **payload,
            editable_paths=editable_paths,
            evaluator_command=evaluator_command,
            objective_direction=objective_direction,
            public_feedback_metrics=public_feedback_metrics,
            metric_patterns=metric_patterns,
            final_holdout_command=final_holdout_command,
            preferred_backend=preferred_backend,
        )


@dataclass(frozen=True)
class FrameworkSpec:
    framework_id: FrameworkKind
    adapter: str
    prompt_profile: str
    edit_mode: str

    def __post_init__(self) -> None:
        if self.edit_mode not in {"direct_workspace", "search_replace_diff"}:
            raise ValueError("unsupported framework edit mode")

    @classmethod
    def from_toml(cls, path: str | Path) -> FrameworkSpec:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        framework_id = FrameworkKind(payload.pop("framework_id"))
        return cls(**payload, framework_id=framework_id)


@dataclass(frozen=True)
class RunAssignment:
    block: int
    order: int
    condition: Condition
    run_seed: int
    run_id: str


def make_assignments(
    spec: FactorialSpec, *, task_id: str, framework_id: str
) -> tuple[RunAssignment, ...]:
    """Create blocked assignments; every block contains each condition once.

    All four cells in a block use the same run seed.  Order is independently
    shuffled per block from the frozen study seed.
    """

    assignments: list[RunAssignment] = []
    for block in range(1, spec.blocks + 1):
        run_seed = int.from_bytes(
            hashlib.sha256(
                f"{spec.study_seed}:{task_id}:{framework_id}:{block}".encode()
            ).digest()[:8],
            "big",
        )
        conditions = list(Condition)
        random.Random(run_seed).shuffle(conditions)
        for order, condition in enumerate(conditions, start=1):
            assignments.append(
                RunAssignment(
                    block=block,
                    order=order,
                    condition=condition,
                    run_seed=run_seed,
                    run_id=(
                        f"{spec.study_id}-{task_id}-{framework_id}-"
                        f"b{block:02d}-{condition.value.lower()}"
                    ),
                )
            )
    return tuple(assignments)
