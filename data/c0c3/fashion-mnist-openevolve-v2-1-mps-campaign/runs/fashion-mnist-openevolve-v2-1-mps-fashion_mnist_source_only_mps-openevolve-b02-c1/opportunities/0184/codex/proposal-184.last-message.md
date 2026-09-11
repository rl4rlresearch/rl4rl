MECHANISM: Mild label smoothing

HYPOTHESIS: Training with 0.03 label smoothing will exceed 9,311 correct predictions by reducing overconfidence on ambiguous examples without changing model capacity or runtime materially.

INTENDED_EDIT: Apply label smoothing only to the existing training cross-entropy; inference and the verified center-weighted logit ensemble remain unchanged.

EVIDENCE: The 0.15 true-class margin reduced correctness from 9,311 to 9,305, suggesting that further sharpening target-class pressure is harmful; mild smoothing tests the complementary loss-level direction at negligible computational cost.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE