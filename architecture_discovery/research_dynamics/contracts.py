"""Versioned contracts for the research-process two-by-two experiment."""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from study.serialization import content_hash, require_bool, require_int, require_str


class VisibleMemoryPolicy(StrEnum):
    SEQUENTIAL = "sequential"
    PORTFOLIO = "portfolio"


class DeliberationPolicy(StrEnum):
    NEUTRAL_REVIEW = "neutral_review"
    ASSUMPTION_CHALLENGE = "assumption_challenge"


class FrameworkKind(StrEnum):
    AUTORESEARCH = "autoresearch"
    OPENEVOLVE = "openevolve"


class ConditionId(StrEnum):
    RD0 = "RD0"
    RD1 = "RD1"
    RD2 = "RD2"
    RD3 = "RD3"


_TREATMENTS = {
    ConditionId.RD0: (
        VisibleMemoryPolicy.SEQUENTIAL,
        DeliberationPolicy.NEUTRAL_REVIEW,
    ),
    ConditionId.RD1: (
        VisibleMemoryPolicy.SEQUENTIAL,
        DeliberationPolicy.ASSUMPTION_CHALLENGE,
    ),
    ConditionId.RD2: (
        VisibleMemoryPolicy.PORTFOLIO,
        DeliberationPolicy.NEUTRAL_REVIEW,
    ),
    ConditionId.RD3: (
        VisibleMemoryPolicy.PORTFOLIO,
        DeliberationPolicy.ASSUMPTION_CHALLENGE,
    ),
}


@dataclass(frozen=True)
class ProcessCondition:
    condition_id: ConditionId
    memory_policy: VisibleMemoryPolicy
    deliberation_policy: DeliberationPolicy

    @classmethod
    def for_id(cls, condition_id: ConditionId | str) -> "ProcessCondition":
        resolved = ConditionId(condition_id)
        memory, deliberation = _TREATMENTS[resolved]
        return cls(resolved, memory, deliberation)

    def __post_init__(self) -> None:
        if (self.memory_policy, self.deliberation_policy) != _TREATMENTS[
            self.condition_id
        ]:
            raise ValueError("condition does not match the frozen treatment mapping")

    def to_dict(self) -> dict[str, str]:
        return {
            "condition_id": self.condition_id.value,
            "memory_policy": self.memory_policy.value,
            "deliberation_policy": self.deliberation_policy.value,
        }


@dataclass(frozen=True)
class ProcessStudyConfig:
    """Frozen controller-visible treatment configuration for one run."""

    study_id: str
    run_id: str
    framework: FrameworkKind
    condition: ProcessCondition
    challenge_opportunities: tuple[int, ...]
    portfolio_size: int = 4
    memory_budget_chars: int = 6000
    source_checkpoint_id: str | None = None
    source_checkpoint_hash: str | None = None
    scientific: bool = False

    SCHEMA_NAME: ClassVar[str] = "ResearchProcessStudyConfig"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        for name in ("study_id", "run_id"):
            value = require_str(getattr(self, name), name)
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{name} must be non-empty and contain no whitespace")
        require_bool(self.scientific, "scientific")
        require_int(self.portfolio_size, "portfolio_size")
        require_int(self.memory_budget_chars, "memory_budget_chars")
        if self.portfolio_size != 4:
            raise ValueError("the frozen portfolio packet has exactly four slots")
        if self.memory_budget_chars < 1000:
            raise ValueError("memory_budget_chars must be at least 1000")
        for index, opportunity in enumerate(self.challenge_opportunities):
            require_int(opportunity, f"challenge_opportunities[{index}]")
            if opportunity < 1:
                raise ValueError("challenge opportunities must be positive")
        if tuple(sorted(set(self.challenge_opportunities))) != self.challenge_opportunities:
            raise ValueError("challenge opportunities must be sorted and unique")
        if self.condition.deliberation_policy is DeliberationPolicy.NEUTRAL_REVIEW:
            if self.challenge_opportunities:
                raise ValueError("neutral-review conditions cannot schedule challenges")
        elif not self.challenge_opportunities:
            raise ValueError("assumption-challenge conditions require a fixed schedule")
        if (self.source_checkpoint_id is None) != (
            self.source_checkpoint_hash is None
        ):
            raise ValueError("checkpoint id and hash must be supplied together")
        if self.source_checkpoint_hash is not None:
            digest = self.source_checkpoint_hash
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("source_checkpoint_hash must be lowercase SHA-256")

    @property
    def config_hash(self) -> str:
        return content_hash(self.to_dict())

    def challenge_active(self, opportunity: int) -> bool:
        return (
            self.condition.deliberation_policy
            is DeliberationPolicy.ASSUMPTION_CHALLENGE
            and opportunity in self.challenge_opportunities
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "framework": self.framework.value,
            "condition": self.condition.to_dict(),
            "challenge_opportunities": list(self.challenge_opportunities),
            "portfolio_size": self.portfolio_size,
            "memory_budget_chars": self.memory_budget_chars,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "scientific": self.scientific,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessStudyConfig":
        expected = {
            "schema_name",
            "schema_version",
            "study_id",
            "run_id",
            "framework",
            "condition",
            "challenge_opportunities",
            "portfolio_size",
            "memory_budget_chars",
            "source_checkpoint_id",
            "source_checkpoint_hash",
            "scientific",
        }
        if set(payload) != expected:
            raise ValueError("research-process config fields differ from schema")
        if payload["schema_name"] != cls.SCHEMA_NAME:
            raise ValueError("expected ResearchProcessStudyConfig schema")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported research-process config schema version")
        raw_condition = payload["condition"]
        if not isinstance(raw_condition, dict):
            raise ValueError("condition must be an object")
        condition = ProcessCondition.for_id(raw_condition.get("condition_id", ""))
        if raw_condition != condition.to_dict():
            raise ValueError("condition fields differ from frozen mapping")
        raw_schedule = payload["challenge_opportunities"]
        if not isinstance(raw_schedule, list):
            raise ValueError("challenge_opportunities must be a list")
        source_id = payload["source_checkpoint_id"]
        source_hash = payload["source_checkpoint_hash"]
        if source_id is not None:
            source_id = require_str(source_id, "source_checkpoint_id")
        if source_hash is not None:
            source_hash = require_str(source_hash, "source_checkpoint_hash")
        return cls(
            study_id=require_str(payload["study_id"], "study_id"),
            run_id=require_str(payload["run_id"], "run_id"),
            framework=FrameworkKind(require_str(payload["framework"], "framework")),
            condition=condition,
            challenge_opportunities=tuple(
                require_int(value, "challenge opportunity") for value in raw_schedule
            ),
            portfolio_size=require_int(payload["portfolio_size"], "portfolio_size"),
            memory_budget_chars=require_int(
                payload["memory_budget_chars"], "memory_budget_chars"
            ),
            source_checkpoint_id=source_id,
            source_checkpoint_hash=source_hash,
            scientific=require_bool(payload["scientific"], "scientific"),
        )


ENV_CONFIG_PATH = "RL4RL_PROCESS_CONFIG"


def config_from_environment() -> ProcessStudyConfig | None:
    """Load an opt-in process config. Existing runs remain byte-for-byte unchanged."""

    raw_path = os.environ.get(ENV_CONFIG_PATH)
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser().resolve()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{ENV_CONFIG_PATH} must point to a JSON object")
    return ProcessStudyConfig.from_dict(payload)


def resolve_initial_candidate(
    default: str | Path,
    *,
    explicit: str | Path | None = None,
    expected_framework: FrameworkKind,
) -> Path:
    """Resolve a fork checkpoint and bind it to the frozen process config hash."""

    default_path = Path(default).expanduser().resolve()
    environment_value = os.environ.get("RL4RL_PROCESS_INITIAL_CANDIDATE")
    if explicit is not None and environment_value is not None:
        if Path(explicit).expanduser().resolve() != Path(environment_value).expanduser().resolve():
            raise ValueError("explicit and environment initial candidates differ")
    selected = Path(explicit or environment_value or default_path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"initial candidate is not a file: {selected}")
    if selected == default_path:
        return selected
    config = config_from_environment()
    if config is None:
        raise ValueError("alternate initial candidate requires RL4RL_PROCESS_CONFIG")
    if config.framework is not expected_framework:
        raise ValueError("alternate initial candidate framework does not match process config")
    if config.source_checkpoint_hash is None:
        raise ValueError("alternate initial candidate requires a frozen checkpoint hash")
    actual = hashlib.sha256(selected.read_bytes()).hexdigest()
    if actual != config.source_checkpoint_hash:
        raise ValueError("initial candidate differs from the randomized checkpoint")
    return selected
