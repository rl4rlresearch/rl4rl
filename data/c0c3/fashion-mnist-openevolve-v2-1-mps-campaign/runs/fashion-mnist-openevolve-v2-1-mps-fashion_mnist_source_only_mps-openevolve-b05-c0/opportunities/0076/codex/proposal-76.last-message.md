MECHANISM: Alternating horizontal-reflection augmentation

HYPOTHESIS: Training every other batch on horizontally reflected images will exceed 9,258 correct predictions while retaining the single-pass evaluation and finishing within the verification limit.

INTENDED_EDIT: Apply a horizontal flip to alternating training batches without changing the model, optimizer, EMA, or calibrated evaluation logits.

EVIDENCE: The original-image/reflection evaluation ensemble targeted useful invariance but timed out because it doubled evaluation inference; training-time reflection introduces that invariance with one forward pass per example and minimal overhead.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del total_steps
    if step % 2 == 1:
        images = images.flip(-1)
    return images, labels
>>>>>>> REPLACE