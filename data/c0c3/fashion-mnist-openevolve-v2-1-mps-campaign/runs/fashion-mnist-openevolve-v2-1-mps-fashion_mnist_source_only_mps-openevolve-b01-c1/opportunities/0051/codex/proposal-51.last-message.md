MECHANISM: Moderate hierarchical flip-pair supervision

HYPOTHESIS: Increasing flip-pair allocation from 25% to 37.5% will exceed 9,245 correct predictions by strengthening the aggregation signal that produced the current best result without fully committing to the unresolved 50% setting.

INTENDED_EDIT: Rebalance the intermediate view loss to 62.5% individual-view and 37.5% flip-pair cross-entropy, preserving all other architecture, optimization, augmentation, and ensemble-loss settings.

EVIDENCE: Adding 25% flip-pair supervision improved correctness from 9,239 to 9,245, while the 50% experiment timed out without accuracy evidence; testing the midpoint is the most direct dose-response follow-up.

<<<<<<< SEARCH
    view_loss = 0.75 * individual_loss + 0.25 * pair_loss
=======
    view_loss = 0.625 * individual_loss + 0.375 * pair_loss
>>>>>>> REPLACE