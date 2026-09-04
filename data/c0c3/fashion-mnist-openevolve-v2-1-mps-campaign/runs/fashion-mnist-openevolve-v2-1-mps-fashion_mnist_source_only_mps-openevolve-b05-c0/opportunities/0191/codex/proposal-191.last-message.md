MECHANISM: Half-batch cyclic translation augmentation

HYPOTHESIS: Training on one-pixel cardinal translations for half of each batch will increase validation_correct above 9,260 by improving spatial robustness without the semantic distortion of horizontal reflection or the evaluation cost of translation ensembling.

INTENDED_EDIT: Apply a deterministic, cyclic one-pixel translation with replicated boundaries to alternating training examples while leaving labels, model, optimizer, EMA, and evaluation calibration unchanged.

EVIDENCE: Horizontal-reflection ensembling reduced validation_correct to 8,877, while translation ensembling was identified as class-preserving but repeatedly exceeded the time limit; moving conservative translation exposure into the existing training forward pass tests the same invariance without extra validation forwards.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del total_steps
    shifts = ((0, 1), (2, 1), (1, 0), (1, 2))
    top, left = shifts[step % len(shifts)]
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    shifted = padded[:, :, top : top + 28, left : left + 28]
    images = images.clone()
    images[::2] = shifted[::2]
    return images, labels
>>>>>>> REPLACE