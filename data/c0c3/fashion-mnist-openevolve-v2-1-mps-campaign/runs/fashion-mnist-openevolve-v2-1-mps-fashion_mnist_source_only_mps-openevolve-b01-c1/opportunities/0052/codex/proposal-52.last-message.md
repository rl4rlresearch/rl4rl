MECHANISM: Refined hierarchical flip-pair supervision

HYPOTHESIS: Increasing flip-pair allocation from 37.5% to 43.75% will exceed 9,246 correct predictions by continuing the observed benefit of stronger validation-aligned aggregation supervision without reaching the unresolved 50% endpoint.

INTENDED_EDIT: Rebalance the view objective to 56.25% individual-view and 43.75% flip-pair cross-entropy while preserving all other settings.

EVIDENCE: Raising flip-pair allocation from 25% to 37.5% improved validation correctness from 9,245 to 9,246; testing the midpoint between 37.5% and the unverified 50% setting is the most direct dose-response follow-up.

<<<<<<< SEARCH
    view_loss = 0.625 * individual_loss + 0.375 * pair_loss
=======
    view_loss = 0.5625 * individual_loss + 0.4375 * pair_loss
>>>>>>> REPLACE