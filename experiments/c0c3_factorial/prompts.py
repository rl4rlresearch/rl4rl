"""Single-source prompt composition with auditable treatment slots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .spec import Condition, FactorialSpec, FrameworkSpec, TaskSpec

SEARCH_SLOT_OPEN = "<!-- TREATMENT:SEARCH_STATE:BEGIN -->"
SEARCH_SLOT_CLOSE = "<!-- TREATMENT:SEARCH_STATE:END -->"
POLICY_SLOT_OPEN = "<!-- TREATMENT:PROPOSAL_POLICY:BEGIN -->"
POLICY_SLOT_CLOSE = "<!-- TREATMENT:PROPOSAL_POLICY:END -->"


@dataclass(frozen=True)
class VisibleCandidate:
    candidate_id: str
    fitness: float
    metrics: dict[str, float | int | str | bool | None]
    selected_count: int
    artifact_path: str
    hypothesis: str = ""


@dataclass(frozen=True)
class PromptContext:
    condition: Condition
    opportunity: int
    selected_parent_id: str
    visible_candidates: tuple[VisibleCandidate, ...]
    remaining_proposals: int
    remaining_evaluations: int
    remaining_tokens: int
    remaining_evaluator_seconds: float


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    common_template_sha256: str
    search_state_sha256: str
    proposal_policy_sha256: str
    prompt_sha256: str
    transition_active: bool


class PromptRenderer:
    """Render C0-C3 from one common template and exactly two variable slots."""

    def __init__(self, template_root: str | Path) -> None:
        root = Path(template_root)
        self.common_template = (root / "common.md").read_text(encoding="utf-8")
        self.ordinary = (root / "ordinary.md").read_text(encoding="utf-8").strip()
        self.transition = (
            (root / "assumption_changing.md").read_text(encoding="utf-8").strip()
        )
        required = {
            "{search_state}",
            "{proposal_policy}",
            "{task_contract}",
            "{framework_contract}",
            "{opportunity}",
            "{selected_parent_id}",
            "{candidate_slots}",
            "{budget_status}",
        }
        missing = [token for token in required if token not in self.common_template]
        if missing:
            raise ValueError(f"common prompt template is missing {missing}")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _task_contract(task: TaskSpec) -> str:
        qualification = (
            "none"
            if task.qualification_metric is None
            else f"{task.qualification_metric} >= {task.qualification_minimum}"
        )
        return (
            f"Task: {task.display_name} ({task.task_id})\n"
            f"Objective: {task.objective_direction.value} {task.objective_metric}\n"
            f"Qualification: {qualification}\n"
            f"Editable paths: {', '.join(task.editable_paths)}\n"
            "Visible evaluation fields: "
            f"{', '.join(task.public_feedback_metrics)}"
        )

    @staticmethod
    def _framework_contract(framework: FrameworkSpec) -> str:
        return (
            f"Framework: {framework.framework_id.value}\n"
            f"Proposal adapter: {framework.adapter}\n"
            f"Edit representation: {'diff' if framework.diff_mode else 'full rewrite'}"
        )

    @staticmethod
    def _search_state(condition: Condition, capacity: int) -> str:
        if condition.has_portfolio:
            return (
                f"Portfolio memory is active. Up to K={capacity} qualified branches "
                "are visible. The controller selects one parent using the frozen rule; "
                "use all visible branches as evidence, but edit only the selected "
                "parent's copy."
            )
        return (
            "Single-incumbent state is active. Exactly one qualified candidate is "
            "visible and it is the selected parent. No rejected or alternate branch "
            "is available."
        )

    @staticmethod
    def _slots(candidates: tuple[VisibleCandidate, ...], capacity: int) -> str:
        if len(candidates) > capacity:
            raise ValueError("visible portfolio exceeds frozen K")
        slots: list[str] = []
        for index in range(capacity):
            if index < len(candidates):
                candidate = candidates[index]
                metrics = json.dumps(candidate.metrics, sort_keys=True)
                slots.append(
                    f"SLOT {index + 1}\n"
                    f"candidate_id: {candidate.candidate_id}\n"
                    f"fitness: {candidate.fitness}\n"
                    f"selected_count: {candidate.selected_count}\n"
                    f"metrics: {metrics}\n"
                    f"prior_hypothesis: {candidate.hypothesis or '[not available]'}\n"
                    f"artifact: {candidate.artifact_path}"
                )
            else:
                slots.append(
                    f"SLOT {index + 1}\n"
                    "candidate_id: [NEUTRAL EMPTY SLOT]\n"
                    "fitness: [NOT AVAILABLE]\n"
                    "selected_count: [NOT AVAILABLE]\n"
                    "metrics: [NOT AVAILABLE]\n"
                    "prior_hypothesis: [NOT AVAILABLE]\n"
                    "artifact: [NOT AVAILABLE]"
                )
        return "\n\n".join(slots)

    def render(
        self,
        spec: FactorialSpec,
        task: TaskSpec,
        framework: FrameworkSpec,
        context: PromptContext,
    ) -> RenderedPrompt:
        transition_active = context.condition.transition_active(
            context.opportunity, spec.transition_opportunities
        )
        search_state = self._search_state(context.condition, spec.portfolio_capacity)
        proposal_policy = self.transition if transition_active else self.ordinary
        budget = (
            f"proposals_remaining={context.remaining_proposals}; "
            f"evaluations_remaining={context.remaining_evaluations}; "
            f"tokens_remaining={context.remaining_tokens}; "
            "evaluator_seconds_remaining="
            f"{context.remaining_evaluator_seconds:.3f}"
        )
        text = self.common_template.format(
            search_state=(f"{SEARCH_SLOT_OPEN}\n{search_state}\n{SEARCH_SLOT_CLOSE}"),
            proposal_policy=(
                f"{POLICY_SLOT_OPEN}\n{proposal_policy}\n{POLICY_SLOT_CLOSE}"
            ),
            task_contract=self._task_contract(task),
            framework_contract=self._framework_contract(framework),
            opportunity=context.opportunity,
            selected_parent_id=context.selected_parent_id,
            candidate_slots=self._slots(
                context.visible_candidates, spec.portfolio_capacity
            ),
            budget_status=budget,
        )
        return RenderedPrompt(
            text=text,
            common_template_sha256=self._hash(self.common_template),
            search_state_sha256=self._hash(search_state),
            proposal_policy_sha256=self._hash(proposal_policy),
            prompt_sha256=self._hash(text),
            transition_active=transition_active,
        )


def treatment_skeleton(prompt: str) -> str:
    """Redact the only two treatment regions for byte-level prompt audits."""

    def redact(text: str, start: str, end: str, label: str) -> str:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"prompt must contain exactly one {label} treatment slot")
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        return f"{prefix}{start}\n[{label} REDACTED]\n{end}{suffix}"

    prompt = redact(prompt, SEARCH_SLOT_OPEN, SEARCH_SLOT_CLOSE, "SEARCH_STATE")
    return redact(prompt, POLICY_SLOT_OPEN, POLICY_SLOT_CLOSE, "PROPOSAL_POLICY")
