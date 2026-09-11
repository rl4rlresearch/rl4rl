MECHANISM: Mild target-distribution smoothing

HYPOTHESIS: Label smoothing of 0.05 will exceed 9,286 correct predictions by adding output-level regularization without changing the proven architecture, augmentation, averaging, or runtime profile.

INTENDED_EDIT: Train with 0.05 label smoothing instead of hard-label cross-entropy.

EVIDENCE: Reducing classifier dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating weaker regularization hurts; label smoothing tests complementary regularization with negligible computational overhead.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE