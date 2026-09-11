MECHANISM: Slightly reduced constant label smoothing

HYPOTHESIS: Constant 0.018 label smoothing will exceed 9,328 correct predictions by retaining the regularization benefit of 0.02 while avoiding the over-regularization observed at 0.025.

INTENDED_EDIT: Train both paired flip views with constant 0.018 label smoothing, leaving the reliable architecture, optimizer, schedule, and confidence-adaptive evaluation fusion unchanged.

EVIDENCE: Constant 0.02 smoothing improved the hard-max baseline from 9,320 to 9,328 correct, while increasing it to 0.025 fell to 9,307 and decaying it toward zero reached 9,325; this motivates a narrow search immediately below the best verified constant value.

<<<<<<< SEARCH
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.018,
    )
>>>>>>> REPLACE