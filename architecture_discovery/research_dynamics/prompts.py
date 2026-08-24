"""Frozen neutral and assumption-challenge prompt interventions."""

from __future__ import annotations


LAB_NOTE_REQUIREMENTS = """Public research-note contract
Encode the following inspectable fields as short JSON strings in the candidate's
top-level metadata. They are descriptive and must not alter executable structure:
- research_current_explanation
- research_evidence
- research_next_experiment
- research_expected_result
- research_decision_rule
- research_previous_interpretation
- research_previous_changed (write exactly \"yes\", \"no\", or \"not_applicable\")
- research_challenged_assumption
- research_alternative_explanation
- research_discriminating_evidence

Do not reveal private chain-of-thought. Each field should be one or two concise,
auditable sentences. The proposed Architecture IR is the experiment named by
research_next_experiment. For the first opportunity, set both previous-result
fields to \"not_applicable\". Never use sealed evaluation or future results.
"""


NEUTRAL_REVIEW = """Deliberation instruction: neutral evidence review
Review the available public results. Choose the next experiment and state which
evidence supports that choice. State the result you expect. Fill the challenged-
assumption fields with \"not_requested\" so the output schema stays constant.
"""


ASSUMPTION_CHALLENGE = """Deliberation instruction: assumption challenge
Identify one claim the current direction treats as true without decisive public
evidence. Select the claim whose rejection would most change the next decision.
State a concrete alternative explanation, what public evidence would favor each
explanation, and choose the next experiment because it distinguishes them. Do
not challenge a cosmetic detail merely to satisfy the instruction.
"""


PLACEBO_AT_UNSCHEDULED_STEP = """Deliberation instruction: scheduled neutral review
No assumption challenge is scheduled at this opportunity. Review the available
public results, state the expected result, and choose the next experiment. Fill
the challenged-assumption fields with \"not_requested\".
"""


def deliberation_block(*, challenge_condition: bool, challenge_active: bool) -> str:
    if challenge_active:
        value = ASSUMPTION_CHALLENGE
    elif challenge_condition:
        value = PLACEBO_AT_UNSCHEDULED_STEP
    else:
        value = NEUTRAL_REVIEW
    # Equalize prompt-character exposure across treatment cells. Token counts
    # are recorded separately because provider tokenizers may split spaces.
    budget = 1800
    if len(value) > budget:
        raise AssertionError("frozen deliberation prompt exceeds its budget")
    return value.ljust(budget)
