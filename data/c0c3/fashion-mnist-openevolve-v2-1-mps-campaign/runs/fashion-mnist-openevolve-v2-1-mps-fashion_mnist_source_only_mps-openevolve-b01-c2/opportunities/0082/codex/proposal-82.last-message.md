MECHANISM: Reduced uniform label smoothing

HYPOTHESIS: Reducing label smoothing from 0.02 to 0.01 will exceed 9,330 correct predictions by improving hard-label class separation while retaining mild regularization.

INTENDED_EDIT: Halve label smoothing for both ensemble and individual-view cross-entropy losses without changing computation, architecture, or scheduling.

EVIDENCE: The best verified design reaches 9,330 correct with fixed 0.02 smoothing; prior hard-label and annealed-smoothing experiments timed out rather than producing contrary validation evidence, motivating a conservative midpoint test.

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