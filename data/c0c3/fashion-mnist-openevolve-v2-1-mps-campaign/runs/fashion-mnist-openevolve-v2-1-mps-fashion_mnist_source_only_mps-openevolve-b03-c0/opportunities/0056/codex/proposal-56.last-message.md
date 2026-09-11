MECHANISM: Moderately reduced label smoothing

HYPOTHESIS: Reducing label smoothing from 0.04 to 0.03 will exceed 9,319 correct predictions by strengthening decision margins while retaining most of the baseline’s regularization.

INTENDED_EDIT: Use fixed 0.03 label smoothing while preserving the verified architecture, optimizer, schedule, and 0.80 evaluation temperature.

EVIDENCE: Evaluation temperatures from 0.95 through 0.80 preserved exactly 9,319 predictions while progressively lowering cross-entropy, indicating underconfident logits; the 0.02 smoothing attempt timed out without contrary accuracy evidence, motivating a more conservative intermediate reduction.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE