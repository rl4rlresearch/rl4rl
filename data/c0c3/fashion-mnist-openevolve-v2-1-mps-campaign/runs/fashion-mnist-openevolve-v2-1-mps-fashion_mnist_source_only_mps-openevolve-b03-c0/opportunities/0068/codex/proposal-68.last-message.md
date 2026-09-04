MECHANISM: Alternating-batch horizontal-reflection augmentation

HYPOTHESIS: Training on horizontally reflected images for half of the fixed exposure will exceed 9,319 correct predictions by learning reflection invariance without the padding artifacts observed with translation ensembling.

INTENDED_EDIT: Horizontally flip every other training batch while preserving the architecture, optimizer, example count, and calibrated evaluation temperature.

EVIDENCE: The translation ensemble lost 19 correct predictions, implicating shifted-boundary artifacts, while reflection-based training timed out without producing contrary accuracy evidence; alternating whole batches tests reflection invariance with minimal runtime overhead.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del total_steps
    if step % 2 == 1:
        images = images.flip(-1)
    return images, labels
>>>>>>> REPLACE