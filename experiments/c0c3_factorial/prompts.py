"""Single-source prompt composition with auditable policy slots."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .neutral_task import (
    ARTIFACT_CLEAN_ASSUMPTION_PROMPT_PATHS,
    ARTIFACT_CLEAN_PROMPT_PROFILES,
    AUTORESEARCH_V17_PROMPT_PROFILE,
    FASHION_MNIST_AUTORESEARCH_V17_PROMPT_PROFILE,
    FASHION_MNIST_OPENEVOLVE_V21_PROMPT_PROFILE,
    NANOGPT_AUTORESEARCH_V17_PROMPT_PROFILE,
    NANOGPT_OPENEVOLVE_V21_PROMPT_PROFILE,
    OPENEVOLVE_V2_PROMPT_PROFILE,
    OPENEVOLVE_V21_PROMPT_PROFILE,
    OPERATOR_PROMPT_ROOT_ENV,
    SUBJECT_NEUTRAL_PROMPT_PROFILES,
    TINY_ADDERBOARD_OPENEVOLVE_V4_PROMPT_PROFILE,
    TINY_KWS_RNN_OPENEVOLVE_V21_PROMPT_PROFILE,
)
from .neutral_task import NEUTRAL_PROMPT_PROFILE as _NEUTRAL_PROMPT_PROFILE
from .spec import (
    Condition,
    ConversationMode,
    FactorialSpec,
    FrameworkKind,
    FrameworkSpec,
    TaskSpec,
)

SEARCH_SLOT_OPEN = "<!-- TREATMENT:SEARCH_STATE:BEGIN -->"
SEARCH_SLOT_CLOSE = "<!-- TREATMENT:SEARCH_STATE:END -->"
POLICY_SLOT_OPEN = "<!-- TREATMENT:PROPOSAL_POLICY:BEGIN -->"
POLICY_SLOT_CLOSE = "<!-- TREATMENT:PROPOSAL_POLICY:END -->"

# Backward-compatible public location used by existing analysis/tests.
NEUTRAL_PROMPT_PROFILE = _NEUTRAL_PROMPT_PROFILE

NEUTRAL_SEARCH_SLOT_OPEN = "<!-- DESIGN_CONTEXT:BEGIN -->"
NEUTRAL_SEARCH_SLOT_CLOSE = "<!-- DESIGN_CONTEXT:END -->"
NEUTRAL_POLICY_SLOT_OPEN = "<!-- NEXT_STEP_GUIDANCE:BEGIN -->"
NEUTRAL_POLICY_SLOT_CLOSE = "<!-- NEXT_STEP_GUIDANCE:END -->"
NEUTRAL_DISCLOSURE_TERMS = (
    "adderboard",
    "benchmark",
    "experiment",
    "study",
    "factorial",
    "treatment",
    "pre-registered",
    "protocol",
    "condition",
    "layer a",
    "controller",
    "c0",
    "c1",
    "c2",
    "c3",
)

FROZEN_ASSUMPTION_PROMPT = Path("subject-prompt/assumption_changing.md")
FROZEN_ASSUMPTION_PROMPT_MANIFEST = Path("subject-prompt/manifest.json")


def artifact_clean_assumption_prompt_source(
    *, campaign: Path, repo_root: Path, framework: FrameworkSpec
) -> Path | None:
    """Resolve the operator-editable prompt used when a trajectory first starts."""

    relative = ARTIFACT_CLEAN_ASSUMPTION_PROMPT_PATHS.get(framework.prompt_profile)
    if relative is None:
        return None
    roots: list[Path] = []
    configured = os.environ.get(OPERATOR_PROMPT_ROOT_ENV)
    if configured:
        roots.append(Path(configured).expanduser().resolve())
    roots.extend(
        parent / "experiments/c0c3_factorial/templates"
        for parent in campaign.resolve().parents
    )
    roots.append(repo_root / "experiments/c0c3_factorial/templates")
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        source = root / relative
        if source.is_file() and not source.is_symlink():
            return source
    raise FileNotFoundError(
        f"operator assumption prompt is missing for {framework.prompt_profile}: "
        f"{relative}"
    )


@dataclass(frozen=True)
class VisibleCandidate:
    candidate_id: str
    fitness: float
    metrics: dict[str, float | int | str | bool | None]
    selected_count: int
    artifact_path: str
    hypothesis: str = ""


@dataclass(frozen=True)
class VisibleOutcome:
    opportunity: int
    hypothesis: str
    intended_edit: str
    metrics: dict[str, float | int | str | bool | None]
    valid: bool
    retained: bool
    failure_kind: str | None
    mechanism: str = "[not recorded]"
    evidence: str = "[not recorded]"
    developmental_status: str | None = None
    developmental_credit: float | None = None
    developmental_reasons: tuple[str, ...] = ()


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
    hide_token_budget: bool = False
    token_budget_continuation_notice: bool = False
    no_search: bool = False
    recent_outcomes: tuple[VisibleOutcome, ...] = ()
    mechanism_ledger: str = "No earlier mechanism result is available."
    phased_session: bool = False
    proposal_policy_override: str | None = None
    evaluation_policy_note: str | None = None


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    common_template_sha256: str
    search_state_sha256: str
    proposal_policy_sha256: str
    prompt_sha256: str
    transition_active: bool
    treatment_skeleton_sha256: str = ""


class PromptRenderer:
    """Render all cells from one common template and two variable slots."""

    def __init__(
        self,
        template_root: str | Path,
        *,
        artifact_clean_transition_override: str | Path | None = None,
    ) -> None:
        root = Path(template_root)
        transition_override = (
            Path(artifact_clean_transition_override)
            if artifact_clean_transition_override is not None
            else None
        )
        override_text = (
            transition_override.read_text(encoding="utf-8").strip()
            if transition_override is not None
            else None
        )
        self.common_template = (root / "common.md").read_text(encoding="utf-8")
        self.ordinary = (root / "ordinary.md").read_text(encoding="utf-8").strip()
        self.transition = (
            (root / "assumption_changing.md").read_text(encoding="utf-8").strip()
        )
        self.continuous_autoresearch_transition = (
            (root / "assumption_changing_continuous_autoresearch.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        neutral_root = root / "transformer_optimizer_v1_5"
        self.neutral_common_template = (neutral_root / "PROGRAM.md").read_text(
            encoding="utf-8"
        )
        self.neutral_transition = (
            (neutral_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        openevolve_root = root / "transformer_optimizer_openevolve_v2"
        self.openevolve_v2_common_template = (openevolve_root / "PROGRAM.md").read_text(
            encoding="utf-8"
        )
        self.openevolve_v2_transition = (
            (openevolve_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        autoresearch_v17_root = root / "transformer_optimizer_v1_7"
        self.autoresearch_v17_initial_template = (
            autoresearch_v17_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.autoresearch_v17_continue_template = (
            autoresearch_v17_root / "CONTINUE.md"
        ).read_text(encoding="utf-8")
        self.autoresearch_v17_transition = (
            override_text
            if override_text is not None
            else (autoresearch_v17_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        openevolve_v21_root = root / "transformer_optimizer_openevolve_v2_1"
        self.openevolve_v21_common_template = (
            openevolve_v21_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.openevolve_v21_transition = (
            override_text
            if override_text is not None
            else (openevolve_v21_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        nanogpt_autoresearch_root = root / "nanogpt_optimizer_v1_7"
        self.nanogpt_autoresearch_v17_initial_template = (
            nanogpt_autoresearch_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.nanogpt_autoresearch_v17_continue_template = (
            nanogpt_autoresearch_root / "CONTINUE.md"
        ).read_text(encoding="utf-8")
        self.nanogpt_autoresearch_v17_transition = (
            override_text
            if override_text is not None
            else (nanogpt_autoresearch_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        nanogpt_openevolve_root = root / "nanogpt_optimizer_openevolve_v2_1"
        self.nanogpt_openevolve_v21_common_template = (
            nanogpt_openevolve_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.nanogpt_openevolve_v21_transition = (
            override_text
            if override_text is not None
            else (nanogpt_openevolve_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        fashion_autoresearch_root = root / "fashion_mnist_optimizer_v1_7"
        self.fashion_autoresearch_v17_initial_template = (
            fashion_autoresearch_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.fashion_autoresearch_v17_continue_template = (
            fashion_autoresearch_root / "CONTINUE.md"
        ).read_text(encoding="utf-8")
        self.fashion_autoresearch_v17_transition = (
            override_text
            if override_text is not None
            else (fashion_autoresearch_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        fashion_openevolve_root = root / "fashion_mnist_optimizer_openevolve_v2_1"
        self.fashion_openevolve_v21_common_template = (
            fashion_openevolve_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.fashion_openevolve_v21_transition = (
            override_text
            if override_text is not None
            else (fashion_openevolve_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        tiny_adderboard_openevolve_root = root / "tiny_adderboard_openevolve_v4"
        self.tiny_adderboard_openevolve_v4_common_template = (
            tiny_adderboard_openevolve_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.tiny_adderboard_openevolve_v4_transition = (
            override_text
            if override_text is not None
            else (tiny_adderboard_openevolve_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        tiny_kws_openevolve_root = root / "tiny_kws_rnn_openevolve_v2_1"
        self.tiny_kws_openevolve_v21_common_template = (
            tiny_kws_openevolve_root / "PROGRAM.md"
        ).read_text(encoding="utf-8")
        self.tiny_kws_openevolve_v21_transition = (
            override_text
            if override_text is not None
            else (tiny_kws_openevolve_root / "assumption_changing.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        self._require_tokens(
            self.common_template,
            {
                "{search_state}",
                "{proposal_policy}",
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{opportunity}",
                "{selected_parent_id}",
                "{candidate_slots}",
                "{budget_status}",
            },
        )
        self._require_tokens(
            self.neutral_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance}",
                "{opportunity}",
                "{budget_status}",
            },
        )
        self._require_tokens(
            self.openevolve_v2_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{mechanism_ledger}",
                "{proposal_guidance}",
                "{opportunity}",
                "{budget_status}",
            },
        )
        self._require_tokens(
            self.autoresearch_v17_initial_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.autoresearch_v17_continue_template,
            {
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.openevolve_v21_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.nanogpt_autoresearch_v17_initial_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.nanogpt_autoresearch_v17_continue_template,
            {
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.nanogpt_openevolve_v21_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.fashion_autoresearch_v17_initial_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{conversation_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.fashion_autoresearch_v17_continue_template,
            {
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.fashion_openevolve_v21_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.tiny_adderboard_openevolve_v4_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )
        self._require_tokens(
            self.tiny_kws_openevolve_v21_common_template,
            {
                "{task_contract}",
                "{framework_contract}",
                "{design_context}",
                "{recent_outcomes}",
                "{proposal_guidance_section}",
            },
        )

    @staticmethod
    def _require_tokens(template: str, required: set[str]) -> None:
        missing = [token for token in required if token not in template]
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
    def _neutral_task_contract(task: TaskSpec) -> str:
        qualification = (
            "No additional accuracy threshold."
            if task.qualification_metric is None
            else (
                f"Required result: {task.qualification_metric} >= "
                f"{task.qualification_minimum}."
            )
        )
        return (
            f"{task.objective_direction.value.capitalize()} "
            f"{task.objective_metric}. {qualification}\n"
            f"Editable source files: {', '.join(task.editable_paths)}.\n"
            "Results reported after each verification: "
            f"{', '.join(task.public_feedback_metrics)}."
        )

    @classmethod
    def _neutral_task_contract_with_policy(
        cls, task: TaskSpec, context: PromptContext
    ) -> str:
        contract = cls._neutral_task_contract(task)
        if context.evaluation_policy_note:
            contract += f"\n{context.evaluation_policy_note.strip()}"
        return contract

    @staticmethod
    def _framework_contract(framework: FrameworkSpec) -> str:
        return (
            f"Framework: {framework.framework_key}\n"
            f"Proposal adapter: {framework.adapter}\n"
            f"Edit representation: {framework.edit_mode}"
        )

    @staticmethod
    def _neutral_framework_contract(framework: FrameworkSpec) -> str:
        if framework.framework_id in {
            FrameworkKind.GREEDY_OPENEVOLVE,
            FrameworkKind.NATIVE_OPENEVOLVE,
        }:
            return (
                "Propose changes through exact SEARCH/REPLACE blocks. The patching "
                "interface applies them to the supplied editable source."
            )
        return (
            "Edit the current source tree directly. Leave the one finished change "
            "in the workspace for the verification process."
        )

    @staticmethod
    def _conversation_contract(spec: FactorialSpec) -> str:
        if spec.conversation_mode is ConversationMode.CONTINUOUS:
            return (
                "This is one persistent Codex conversation for this run. Before each "
                "opportunity the controller refreshes the workspace with the selected "
                "parent; treat the current filesystem and the structured Layer A state "
                "below as authoritative. Conversation history is retained in every "
                "condition."
            )
        return (
            "This opportunity uses a fresh, ephemeral Codex conversation. No prior "
            "conversation transcript is available."
        )

    @staticmethod
    def _neutral_conversation_contract(
        spec: FactorialSpec, *, phased_session: bool = False
    ) -> str:
        if phased_session:
            return (
                "This working conversation covers a short research phase. The "
                "current filesystem, available-design section, and verification "
                "evidence are authoritative whenever they differ from an earlier "
                "message. A later phase may begin with a fresh conversation."
            )
        if spec.conversation_mode is ConversationMode.CONTINUOUS:
            return (
                "This working conversation continues across cycles. The current "
                "filesystem, available-design section, and recent verification "
                "section are authoritative whenever they differ from older messages."
            )
        return (
            "This is a fresh working conversation. Use the current filesystem and "
            "the information below as the complete available context."
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
    def _neutral_search_state(condition: Condition, capacity: int) -> str:
        if condition.has_portfolio:
            return (
                f"Up to {capacity} qualified designs are available as read-only "
                "references. One is loaded as the current editable design. Use every "
                "provided design as technical evidence, but edit only the current "
                "workspace."
            )
        return (
            "One qualified design is loaded as the current editable design. No "
            "alternative design is provided in this cycle."
        )

    @staticmethod
    def _clean_search_state(condition: Condition) -> str:
        if condition.has_portfolio:
            return (
                "The current editable design and the qualified reference designs "
                "below are available as technical evidence. Edit only the current "
                "workspace."
            )
        return (
            "The current editable design is provided. No reference design is available."
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

    @staticmethod
    def _neutral_slots(
        candidates: tuple[VisibleCandidate, ...],
        capacity: int,
        selected_parent_id: str,
    ) -> str:
        if len(candidates) > capacity:
            raise ValueError("visible portfolio exceeds configured capacity")
        slots: list[str] = []
        for index in range(capacity):
            if index < len(candidates):
                candidate = candidates[index]
                metrics = json.dumps(candidate.metrics, sort_keys=True)
                status = (
                    "current editable design"
                    if candidate.candidate_id == selected_parent_id
                    else "read-only reference"
                )
                slots.append(
                    f"DESIGN {index + 1}\n"
                    f"status: {status}\n"
                    f"verified_results: {metrics}\n"
                    f"times_used_as_starting_point: {candidate.selected_count}\n"
                    f"prior_hypothesis: {candidate.hypothesis or '[not available]'}\n"
                    f"source: {candidate.artifact_path}"
                )
            else:
                slots.append(
                    f"DESIGN {index + 1}\n"
                    "status: [no design provided]\n"
                    "verified_results: [not available]\n"
                    "times_used_as_starting_point: [not available]\n"
                    "prior_hypothesis: [not available]\n"
                    "source: [not available]"
                )
        return "\n\n".join(slots)

    @staticmethod
    def _neutral_recent_outcomes(outcomes: tuple[VisibleOutcome, ...]) -> str:
        if not outcomes:
            return "No earlier verification result is available."
        rows: list[str] = []
        for outcome in outcomes:
            if outcome.valid and outcome.retained:
                result = "met the accuracy requirement and became an available design"
            elif outcome.valid:
                result = "met the accuracy requirement but was not a strict improvement"
            elif outcome.failure_kind == "nonqualification":
                result = "did not meet the accuracy requirement"
            else:
                reason = outcome.failure_kind or "unknown error"
                result = f"could not be verified ({reason})"
            rows.append(
                f"WORK CYCLE {outcome.opportunity}\n"
                f"hypothesis: {outcome.hypothesis}\n"
                f"change: {outcome.intended_edit}\n"
                f"mechanism: {outcome.mechanism}\n"
                f"evidence_used: {outcome.evidence}\n"
                f"result: {result}\n"
                f"reported_values: {json.dumps(outcome.metrics, sort_keys=True)}"
            )
        return "\n\n".join(rows)

    @staticmethod
    def _public_metrics(
        metrics: dict[str, float | int | str | bool | None],
        task: TaskSpec,
    ) -> dict[str, float | int | str | bool | None]:
        return {
            name: metrics[name]
            for name in task.public_feedback_metrics
            if name in metrics
        }

    @classmethod
    def _clean_slots(
        cls,
        candidates: tuple[VisibleCandidate, ...],
        *,
        selected_parent_id: str,
        task: TaskSpec,
        include_source_paths: bool,
    ) -> str:
        current = [
            candidate
            for candidate in candidates
            if candidate.candidate_id == selected_parent_id
        ]
        if len(current) != 1:
            raise ValueError("available designs must contain the selected design once")
        references = [
            candidate
            for candidate in candidates
            if candidate.candidate_id != selected_parent_id
        ]
        rows: list[str] = []
        for label, candidate in [
            ("CURRENT DESIGN", current[0]),
            *[
                (f"REFERENCE DESIGN {index}", candidate)
                for index, candidate in enumerate(references, start=1)
            ],
        ]:
            metrics = json.dumps(
                cls._public_metrics(candidate.metrics, task), sort_keys=True
            )
            parts = [label, f"verified_results: {metrics}"]
            hypothesis = candidate.hypothesis.strip()
            if hypothesis and not hypothesis.startswith("[missing"):
                if hypothesis == "frozen seed baseline":
                    hypothesis = "starting design"
                parts.append(f"prior_hypothesis: {hypothesis}")
            if include_source_paths and label != "CURRENT DESIGN":
                parts.append(f"source: {candidate.artifact_path}")
            rows.append("\n".join(parts))
        return "\n\n".join(rows)

    @staticmethod
    def _clean_failure_result(outcome: VisibleOutcome) -> str:
        messages = {
            "nonqualification": "did not meet the accuracy requirement",
            "unmatched_diff": "the patch search text did not match the source",
            "ambiguous_diff": "the patch search text matched more than once",
            "empty_search": "the patch contained an empty search section",
            "malformed_diff": "the patch format was malformed",
            "invalid_diff": "the patch could not be applied",
            "source_preflight": "the edited source did not pass submission checks",
            "model_contract": (
                "the trained implementation did not satisfy the learned-model "
                "requirement"
            ),
            "timeout": "training did not finish within the verification time limit",
            "duplicate": "the edit reproduced a previously verified implementation",
            "provider": "no usable edit was produced",
            "infrastructure_interruption": "no usable edit was produced",
        }
        return messages.get(
            outcome.failure_kind or "",
            "the implementation could not be verified",
        )

    @classmethod
    def _clean_recent_outcomes(
        cls,
        outcomes: tuple[VisibleOutcome, ...],
        task: TaskSpec,
    ) -> str:
        if not outcomes:
            return "No earlier verification result is available."
        rows: list[str] = []
        for outcome in outcomes:
            if task.qualification_metric is None and outcome.valid and outcome.retained:
                result = "improved the objective and became an available design"
            elif task.qualification_metric is None and outcome.valid:
                result = "was valid but was not a strict improvement"
            elif outcome.valid and outcome.retained:
                result = "met the accuracy requirement and became an available design"
            elif outcome.valid:
                result = "met the accuracy requirement but was not a strict improvement"
            else:
                result = cls._clean_failure_result(outcome)
            parts = ["RECENT RESULT"]
            for label, value, omitted in (
                ("hypothesis", outcome.hypothesis, "[missing hypothesis]"),
                ("change", outcome.intended_edit, "[missing intended edit]"),
                ("mechanism", outcome.mechanism, "[not recorded]"),
                ("evidence_used", outcome.evidence, "[not recorded]"),
            ):
                normalized = value.strip()
                if normalized and normalized != omitted:
                    parts.append(f"{label}: {normalized}")
            if outcome.developmental_status is not None:
                parts.append(
                    "developmental_status: "
                    f"{outcome.developmental_status}; "
                    f"credit={outcome.developmental_credit}; "
                    f"evidence={', '.join(outcome.developmental_reasons) or 'none'}"
                )
            parts.append(f"result: {result}")
            metrics = cls._public_metrics(outcome.metrics, task)
            if metrics:
                parts.append(f"reported_values: {json.dumps(metrics, sort_keys=True)}")
            rows.append("\n".join(parts))
        return "\n\n".join(rows)

    def render(
        self,
        spec: FactorialSpec,
        task: TaskSpec,
        framework: FrameworkSpec,
        context: PromptContext,
    ) -> RenderedPrompt:
        neutral = framework.prompt_profile in SUBJECT_NEUTRAL_PROMPT_PROFILES
        artifact_clean = framework.prompt_profile in ARTIFACT_CLEAN_PROMPT_PROFILES
        autoresearch_v17 = framework.prompt_profile == AUTORESEARCH_V17_PROMPT_PROFILE
        openevolve_v2 = framework.prompt_profile == OPENEVOLVE_V2_PROMPT_PROFILE
        openevolve_v21 = framework.prompt_profile == OPENEVOLVE_V21_PROMPT_PROFILE
        nanogpt_autoresearch_v17 = (
            framework.prompt_profile == NANOGPT_AUTORESEARCH_V17_PROMPT_PROFILE
        )
        nanogpt_openevolve_v21 = (
            framework.prompt_profile == NANOGPT_OPENEVOLVE_V21_PROMPT_PROFILE
        )
        fashion_autoresearch_v17 = (
            framework.prompt_profile == FASHION_MNIST_AUTORESEARCH_V17_PROMPT_PROFILE
        )
        fashion_openevolve_v21 = (
            framework.prompt_profile == FASHION_MNIST_OPENEVOLVE_V21_PROMPT_PROFILE
        )
        tiny_adderboard_openevolve_v4 = (
            framework.prompt_profile == TINY_ADDERBOARD_OPENEVOLVE_V4_PROMPT_PROFILE
        )
        tiny_kws_openevolve_v21 = (
            framework.prompt_profile == TINY_KWS_RNN_OPENEVOLVE_V21_PROMPT_PROFILE
        )
        transition_active = (
            False
            if context.no_search
            else context.condition.transition_active(
                context.opportunity, spec.transition_opportunities
            )
        )
        if context.no_search:
            if neutral:
                search_state = (
                    "Work from the supplied starting design. No earlier design result "
                    "or alternative implementation is provided."
                )
                proposal_policy = (
                    "Produce the strongest single improvement you can from the "
                    "supplied "
                    "starting design."
                )
            else:
                search_state = (
                    "No-controller-search baseline. Exactly the frozen seed is visible "
                    "in the controller state. This proposal receives no "
                    "controller-supplied previous proposal, evaluation, or adaptive "
                    "search feedback, and its "
                    "result will not alter later controller state."
                )
                proposal_policy = (
                    "Produce one best-effort proposal from the frozen seed supplied by "
                    "the controller. Do not request controller-supplied prior results."
                )
        else:
            search_state = (
                self._clean_search_state(context.condition)
                if artifact_clean
                else (
                    self._neutral_search_state(
                        context.condition, spec.portfolio_capacity
                    )
                    if neutral
                    else self._search_state(context.condition, spec.portfolio_capacity)
                )
            )
            if transition_active:
                if neutral:
                    proposal_policy = (
                        self.nanogpt_autoresearch_v17_transition
                        if nanogpt_autoresearch_v17
                        else (
                            self.nanogpt_openevolve_v21_transition
                            if nanogpt_openevolve_v21
                            else (
                                self.fashion_autoresearch_v17_transition
                                if fashion_autoresearch_v17
                                else (
                                    self.fashion_openevolve_v21_transition
                                    if fashion_openevolve_v21
                                    else (
                                        self.tiny_adderboard_openevolve_v4_transition
                                        if tiny_adderboard_openevolve_v4
                                        else (
                                            self.tiny_kws_openevolve_v21_transition
                                            if tiny_kws_openevolve_v21
                                            else (
                                                self.autoresearch_v17_transition
                                                if autoresearch_v17
                                                else (
                                                    self.openevolve_v21_transition
                                                    if openevolve_v21
                                                    else (
                                                        self.openevolve_v2_transition
                                                        if openevolve_v2
                                                        else self.neutral_transition
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                else:
                    proposal_policy = (
                        self.continuous_autoresearch_transition
                        if spec.conversation_mode is ConversationMode.CONTINUOUS
                        and framework.framework_id is FrameworkKind.AUTORESEARCH
                        else self.transition
                    )
            else:
                proposal_policy = "" if neutral else self.ordinary
            if transition_active and context.proposal_policy_override is not None:
                proposal_policy = context.proposal_policy_override.strip()
        treatment_skeleton_sha256 = ""
        if artifact_clean:
            include_source_paths = (
                autoresearch_v17 or nanogpt_autoresearch_v17 or fashion_autoresearch_v17
            )
            design_context = f"{search_state}\n\n" + self._clean_slots(
                context.visible_candidates,
                selected_parent_id=context.selected_parent_id,
                task=task,
                include_source_paths=include_source_paths,
            )
            guidance_section = (
                f"## Direction\n\n{proposal_policy}" if proposal_policy else ""
            )
            if nanogpt_autoresearch_v17:
                common_template = (
                    self.nanogpt_autoresearch_v17_initial_template
                    if context.opportunity == 1
                    else self.nanogpt_autoresearch_v17_continue_template
                )
            elif fashion_autoresearch_v17:
                common_template = (
                    self.fashion_autoresearch_v17_initial_template
                    if context.opportunity == 1
                    else self.fashion_autoresearch_v17_continue_template
                )
            elif autoresearch_v17:
                common_template = (
                    self.autoresearch_v17_initial_template
                    if context.opportunity == 1
                    else self.autoresearch_v17_continue_template
                )
            elif nanogpt_openevolve_v21:
                common_template = self.nanogpt_openevolve_v21_common_template
            elif fashion_openevolve_v21:
                common_template = self.fashion_openevolve_v21_common_template
            elif tiny_adderboard_openevolve_v4:
                common_template = self.tiny_adderboard_openevolve_v4_common_template
            elif tiny_kws_openevolve_v21:
                common_template = self.tiny_kws_openevolve_v21_common_template
            else:
                common_template = self.openevolve_v21_common_template
            values = {
                "task_contract": self._neutral_task_contract_with_policy(
                    task, context
                ),
                "framework_contract": self._neutral_framework_contract(framework),
                "conversation_contract": self._neutral_conversation_contract(
                    spec, phased_session=context.phased_session
                ),
                "design_context": design_context,
                "recent_outcomes": self._clean_recent_outcomes(
                    () if context.no_search else context.recent_outcomes,
                    task,
                ),
                "proposal_guidance_section": guidance_section,
            }
            text = common_template.format(**values)
            skeleton_values = dict(values)
            skeleton_values["design_context"] = "[DESIGN CONTEXT REDACTED]"
            skeleton_values["proposal_guidance_section"] = "[DIRECTION REDACTED]"
            treatment_skeleton_sha256 = self._hash(
                common_template.format(**skeleton_values)
            )
        elif neutral:
            budget_parts = [
                f"work_cycles={context.remaining_proposals}",
                f"verifications={context.remaining_evaluations}",
            ]
            if not context.hide_token_budget:
                budget_parts.append(f"tokens={context.remaining_tokens}")
            budget_parts.append(
                f"verification_seconds={context.remaining_evaluator_seconds:.3f}"
            )
            budget = "; ".join(budget_parts)
            if context.token_budget_continuation_notice:
                budget += (
                    "\nYou are allowed to continue past the previously stated "
                    "token budget."
                )
            slots = self._neutral_slots(
                context.visible_candidates,
                spec.portfolio_capacity,
                context.selected_parent_id,
            )
            design_context = (
                f"{NEUTRAL_SEARCH_SLOT_OPEN}\n{search_state}\n\n{slots}\n"
                f"{NEUTRAL_SEARCH_SLOT_CLOSE}"
            )
            guidance = (
                f"{NEUTRAL_POLICY_SLOT_OPEN}\n{proposal_policy}\n"
                f"{NEUTRAL_POLICY_SLOT_CLOSE}"
            )
            common_template = (
                self.openevolve_v2_common_template
                if openevolve_v2
                else self.neutral_common_template
            )
            text = common_template.format(
                task_contract=self._neutral_task_contract_with_policy(task, context),
                framework_contract=self._neutral_framework_contract(framework),
                conversation_contract=self._neutral_conversation_contract(
                    spec, phased_session=context.phased_session
                ),
                design_context=design_context,
                recent_outcomes=self._neutral_recent_outcomes(
                    () if context.no_search else context.recent_outcomes
                ),
                mechanism_ledger=context.mechanism_ledger,
                proposal_guidance=guidance,
                opportunity=context.opportunity,
                budget_status=budget,
            )
        else:
            budget_parts = [
                f"proposals_remaining={context.remaining_proposals}",
                f"evaluations_remaining={context.remaining_evaluations}",
            ]
            if not context.hide_token_budget:
                budget_parts.append(f"tokens_remaining={context.remaining_tokens}")
            budget_parts.append(
                f"evaluator_seconds_remaining={context.remaining_evaluator_seconds:.3f}"
            )
            budget = "; ".join(budget_parts)
            if context.token_budget_continuation_notice:
                budget += (
                    "\nYou are allowed to continue past the previously stated "
                    "token budget."
                )
            common_template = self.common_template
            text = common_template.format(
                search_state=(
                    f"{SEARCH_SLOT_OPEN}\n{search_state}\n{SEARCH_SLOT_CLOSE}"
                ),
                proposal_policy=(
                    f"{POLICY_SLOT_OPEN}\n{proposal_policy}\n{POLICY_SLOT_CLOSE}"
                ),
                task_contract=self._task_contract(task),
                framework_contract=self._framework_contract(framework),
                conversation_contract=self._conversation_contract(spec),
                opportunity=context.opportunity,
                selected_parent_id=context.selected_parent_id,
                candidate_slots=self._slots(
                    context.visible_candidates, spec.portfolio_capacity
                ),
                budget_status=budget,
            )
        return RenderedPrompt(
            text=text,
            common_template_sha256=self._hash(common_template),
            search_state_sha256=self._hash(search_state),
            proposal_policy_sha256=self._hash(proposal_policy),
            prompt_sha256=self._hash(text),
            transition_active=transition_active,
            treatment_skeleton_sha256=treatment_skeleton_sha256,
        )


def treatment_skeleton(prompt: str) -> str:
    """Redact the two variable regions for byte-level prompt audits."""

    def redact(text: str, start: str, end: str, label: str) -> str:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"prompt must contain exactly one {label} policy slot")
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        return f"{prefix}{start}\n[{label} REDACTED]\n{end}{suffix}"

    if NEUTRAL_SEARCH_SLOT_OPEN in prompt:
        prompt = redact(
            prompt,
            NEUTRAL_SEARCH_SLOT_OPEN,
            NEUTRAL_SEARCH_SLOT_CLOSE,
            "DESIGN_CONTEXT",
        )
        return redact(
            prompt,
            NEUTRAL_POLICY_SLOT_OPEN,
            NEUTRAL_POLICY_SLOT_CLOSE,
            "NEXT_STEP_GUIDANCE",
        )
    prompt = redact(prompt, SEARCH_SLOT_OPEN, SEARCH_SLOT_CLOSE, "SEARCH_STATE")
    return redact(prompt, POLICY_SLOT_OPEN, POLICY_SLOT_CLOSE, "PROPOSAL_POLICY")


def neutral_disclosure_terms(prompt: str) -> tuple[str, ...]:
    """Return internal-study terms accidentally exposed by a neutral prompt."""

    lowered = prompt.lower()
    return tuple(term for term in NEUTRAL_DISCLOSURE_TERMS if term in lowered)
