"""Blinded annotation exports for process and logic outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_dynamics.codebook import (
    EpistemicPurpose,
    EvidenceResponse,
    ResearchDisplacement,
    ResearchMove,
)
from research_dynamics.memory import read_jsonl
from research_dynamics.extraction import write_jsonl
from study.serialization import atomic_write_json, stable_id


def blinded_packet(decision: dict[str, Any]) -> dict[str, Any]:
    """Remove treatment, framework, final-score, and future-success information."""

    local = {
        "opportunity": decision.get("opportunity"),
        "lab_note_before": decision.get("lab_note_before"),
        "public_result": decision.get("public_result"),
        "retention_decision": decision.get("retention_decision"),
        "lab_note_after": decision.get("lab_note_after"),
    }
    return {
        "schema_name": "BlindedResearchDecisionPacket",
        "schema_version": "1.0",
        "blinded_id": stable_id(
            "blind",
            {
                "run_id": decision.get("run_id"),
                "opportunity": decision.get("opportunity"),
                "record_hash": decision.get("record_hash"),
            },
        ),
        **local,
    }


def annotation_template(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "ResearchDecisionAnnotation",
        "schema_version": "1.0",
        "blinded_id": packet["blinded_id"],
        "annotator_id": "",
        "research_move": "",
        "epistemic_purpose": "",
        "evidence_response": "",
        "research_displacement": "",
        "discriminating_experiment": None,
        "prediction_contradicted": None,
        "rationale_action_aligned": None,
        "interpretation_supported_by_result": None,
        "hypothesis_id": "",
        "notes": "",
    }


def export_blinded_annotations(
    decisions_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    decisions = read_jsonl(decisions_path)
    packets = [blinded_packet(decision) for decision in decisions]
    templates = [annotation_template(packet) for packet in packets]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    packet_path = destination / "blinded_packets.jsonl"
    annotation_path = destination / "annotation_template.jsonl"
    codebook_path = destination / "annotation_codebook.json"
    write_jsonl(packet_path, packets)
    write_jsonl(annotation_path, templates)
    atomic_write_json(
        codebook_path,
        {
            "schema_name": "ResearchDecisionAnnotationCodebook",
            "schema_version": "1.0",
            "research_move": [item.value for item in ResearchMove],
            "epistemic_purpose": [item.value for item in EpistemicPurpose],
            "evidence_response": [item.value for item in EvidenceResponse],
            "research_displacement": [item.value for item in ResearchDisplacement],
            "boolean_fields": [
                "discriminating_experiment",
                "prediction_contradicted",
                "rationale_action_aligned",
                "interpretation_supported_by_result",
            ],
            "instructions": (
                "Annotate each packet from its local public record only. Do not infer "
                "the treatment, framework, final run score, or later success."
            ),
        },
    )
    return packet_path, annotation_path, codebook_path
