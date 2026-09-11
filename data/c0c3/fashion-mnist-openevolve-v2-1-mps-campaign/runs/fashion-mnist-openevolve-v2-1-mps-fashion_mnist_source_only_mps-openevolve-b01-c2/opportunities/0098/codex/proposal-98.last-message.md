MECHANISM: Reduced label-smoothing bias

HYPOTHESIS: Halving label smoothing from 0.02 to 0.01 will exceed 9,330 correct predictions by strengthening class-boundary gradients while retaining mild regularization.

INTENDED_EDIT: Use 0.01 label smoothing for both ensemble and individual-view cross-entropy losses, with no added computation.

EVIDENCE: The current design verifies 9,330 correct at 0.02 smoothing, while evaluation-temperature changes cannot alter correctness and hard-label attempts produced no completed contrary evidence; this tests a conservative intermediate objective at identical runtime.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.01,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.01)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.01)
    )
>>>>>>> REPLACE