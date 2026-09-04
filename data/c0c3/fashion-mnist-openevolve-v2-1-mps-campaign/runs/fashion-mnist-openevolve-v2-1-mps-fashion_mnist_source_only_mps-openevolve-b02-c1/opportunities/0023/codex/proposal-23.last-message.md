MECHANISM: Mild target-distribution smoothing

HYPOTHESIS: Applying 0.05 label smoothing will exceed 9,280 correct predictions by regularizing the fixed two-pass training horizon without adding meaningful runtime.

INTENDED_EDIT: Add mild label smoothing to the existing cross-entropy loss while preserving the proven architecture, batch size, optimizer, schedule, augmentation, and validation ensemble.

EVIDENCE: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating that weaker regularization hurts; label smoothing tests complementary regularization without the runtime risk of larger models, smaller batches, or parameter averaging.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE