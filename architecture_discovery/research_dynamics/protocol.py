"""Opt-in treatment protocol shared by AutoResearch and OpenEvolve."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_dynamics.contracts import (
    DeliberationPolicy,
    FrameworkKind,
    ProcessStudyConfig,
    config_from_environment,
)
from research_dynamics.memory import read_jsonl, render_memory_packet
from research_dynamics.prompts import LAB_NOTE_REQUIREMENTS, deliberation_block
from study.serialization import atomic_write_json, content_hash


class ProcessProtocol:
    """Build prompts and exposure logs while leaving search mechanics unchanged."""

    def __init__(self, config: ProcessStudyConfig, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.directory = self.output_dir / "research_process"
        self.directory.mkdir(parents=True, exist_ok=True)
        frozen_path = self.directory / "study_config.json"
        payload = config.to_dict()
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != payload:
                raise ValueError("research-process config changed within a run")
        else:
            atomic_write_json(frozen_path, payload)

    @classmethod
    def from_environment(
        cls,
        output_dir: str | Path,
        *,
        expected_framework: FrameworkKind,
    ) -> "ProcessProtocol | None":
        config = config_from_environment()
        if config is None:
            return None
        if config.framework is not expected_framework:
            raise ValueError(
                f"process config framework is {config.framework.value}, expected "
                f"{expected_framework.value}"
            )
        return cls(config, output_dir)

    def prompt_block(
        self,
        opportunity: int,
        *,
        lineage_path: str | Path | None = None,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        if history is not None and lineage_path is not None:
            raise ValueError("supply history or lineage_path, not both")
        records = list(history) if history is not None else read_jsonl(lineage_path) if lineage_path else []
        packet, entries = render_memory_packet(
            records,
            self.config.condition.memory_policy,
            budget_chars=self.config.memory_budget_chars,
        )
        challenge_active = self.config.challenge_active(opportunity)
        deliberation = deliberation_block(
            challenge_condition=(
                self.config.condition.deliberation_policy
                is DeliberationPolicy.ASSUMPTION_CHALLENGE
            ),
            challenge_active=challenge_active,
        )
        block = (
            "\n\n# Research-process experiment (controller-visible)\n"
            + packet
            + "\n\n"
            + deliberation
            + "\n\n"
            + LAB_NOTE_REQUIREMENTS
        )
        self._append_exposure(
            {
                "schema_name": "ResearchProcessExposure",
                "schema_version": "1.0",
                "study_id": self.config.study_id,
                "run_id": self.config.run_id,
                "framework": self.config.framework.value,
                "condition_id": self.config.condition.condition_id.value,
                "opportunity": opportunity,
                "challenge_active": challenge_active,
                "memory_policy": self.config.condition.memory_policy.value,
                "memory_entries": entries,
                "memory_packet_chars": len(packet),
                "intervention_block_chars": len(block),
                "intervention_block_hash": content_hash(block),
            }
        )
        return block

    def augment_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        opportunity: int,
        *,
        lineage_path: str | Path | None = None,
    ) -> list[dict[str, str]]:
        copied = [dict(message) for message in messages]
        if not copied or copied[-1].get("role") != "user":
            raise ValueError("expected the final message to be the user proposal prompt")
        copied[-1]["content"] = copied[-1]["content"] + self.prompt_block(
            opportunity, lineage_path=lineage_path
        )
        return copied

    def system_prompt_block(self) -> str:
        """Static OpenEvolve contract; dynamic memory is installed separately."""

        return LAB_NOTE_REQUIREMENTS

    def _append_exposure(self, record: dict[str, Any]) -> None:
        path = self.directory / "exposures.jsonl"
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)


def protocol_environment(config_path: str | Path) -> dict[str, str]:
    env = dict(os.environ)
    env["RL4RL_PROCESS_CONFIG"] = str(Path(config_path).resolve())
    return env
