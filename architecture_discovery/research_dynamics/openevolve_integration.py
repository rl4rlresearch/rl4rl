"""Project-owned prompt instrumentation for OpenEvolve.

The patch is scoped to one controller run and restored afterward. It changes
only prompt construction. Database sampling, evaluation, replacement, and
stopping remain untouched.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from research_dynamics.protocol import ProcessProtocol


def _as_record(program: dict[str, Any], opportunity: int) -> dict[str, Any]:
    code = program.get("code", "")
    hypothesis = program.get("changes_description", "")
    if isinstance(code, str):
        stripped = code.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            stripped = stripped[7:-3].strip()
        try:
            candidate = json.loads(stripped)
        except (TypeError, ValueError):
            candidate = None
        if isinstance(candidate, dict) and isinstance(candidate.get("metadata"), dict):
            hypothesis = candidate["metadata"].get(
                "research_current_explanation",
                candidate["metadata"].get("mechanism_hypothesis", hypothesis),
            )
    metrics = program.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "candidate_id": str(program.get("id", f"history-{opportunity}")),
        "mechanism_hypothesis": str(hypothesis or ""),
        "evaluation": {
            key: metrics[key]
            for key in (
                "execution_ok",
                "transformer_valid",
                "public_accuracy",
                "search_score",
                "eligible_for_parent",
            )
            if key in metrics
        }
        | {"opportunity_index": opportunity},
        "retention_decision": (
            "accept" if metrics.get("eligible_for_parent") else "reject"
        ),
    }


@contextmanager
def instrument_openevolve_prompts(protocol: ProcessProtocol | None) -> Iterator[None]:
    if protocol is None:
        yield
        return

    from openevolve.prompt.sampler import PromptSampler

    original_history = PromptSampler._format_evolution_history
    original_build = PromptSampler.build_prompt
    call_counter = 0

    def controlled_history(
        sampler,
        previous_programs,
        top_programs,
        inspirations,
        language,
        feature_dimensions=None,
    ):
        # The fixed, character-budgeted packet is appended below. Suppress the
        # vendor's separately sized history in both cells so it cannot leak extra
        # portfolio entries into the sequential condition.
        return "The randomized visible-memory packet appears below."

    def controlled_build(sampler, *args, **kwargs):
        nonlocal call_counter
        result = original_build(sampler, *args, **kwargs)
        call_counter += 1
        evolution_round = kwargs.get("evolution_round", 0)
        opportunity = (
            evolution_round
            if isinstance(evolution_round, int) and evolution_round > 0
            else call_counter
        )
        previous = kwargs.get("previous_programs", [])
        top = kwargs.get("top_programs", [])
        inspirations = kwargs.get("inspirations", [])
        combined = []
        for index, program in enumerate([*previous, *top, *inspirations], 1):
            if isinstance(program, dict):
                combined.append(_as_record(program, index))
        result = dict(result)
        result["user"] = result["user"] + protocol.prompt_block(
            opportunity, history=combined
        )
        return result

    PromptSampler._format_evolution_history = controlled_history
    PromptSampler.build_prompt = controlled_build
    try:
        yield
    finally:
        PromptSampler._format_evolution_history = original_history
        PromptSampler.build_prompt = original_build
