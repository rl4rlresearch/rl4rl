MECHANISM: Hard-target cross-entropy for faster finite-budget fitting

HYPOTHESIS: Removing 0.05 label smoothing will exceed 9,170 correct predictions by strengthening the correct-class gradient during the fixed two-pass exposure budget.

INTENDED_EDIT: Retain the successful architecture, optimizer schedule, and tail EMA while changing training to ordinary cross-entropy.

EVIDENCE: The current unaugmented EMA design is best at 9,170 correct, while input transformations sharply regressed; this isolates a loss-level change, and the short 100,000-example budget makes label smoothing’s weaker target signal a plausible source of underfitting.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE