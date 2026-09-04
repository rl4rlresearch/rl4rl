MECHANISM: Refined low-strength label smoothing

HYPOTHESIS: Reducing label smoothing from 0.05 to 0.04 will exceed 9,258 correct predictions by retaining beneficial regularization while slightly reducing target bias, with no added runtime.

INTENDED_EDIT: Change only the cross-entropy label-smoothing coefficient from 0.05 to 0.04.

EVIDENCE: The 0.05 setting outperformed both hard targets and 0.10 smoothing, establishing a favorable low-strength region; refining it locally is runtime-neutral, unlike recent augmentation and architecture changes that timed out.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
>>>>>>> REPLACE