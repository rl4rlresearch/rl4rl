MECHANISM: Stochastic horizontal-reflection augmentation

HYPOTHESIS: Randomly flipping each training image horizontally with probability 0.5 will exceed 9,319 correct predictions by teaching left-right invariance without the harmful boundary shifts observed with translation ensembling.

INTENDED_EDIT: Apply independent per-image horizontal flips in `prepare_training_batch`; preserve the architecture, optimizer, loss, schedule, and verified 0.80 evaluation temperature.

EVIDENCE: Temperature calibration repeatedly left all 9,319 argmax predictions unchanged, so accuracy gains require altered training; evaluation-time horizontal translations reduced accuracy to 9,300, motivating a semantic reflection augmentation that avoids translated padding artifacts.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
>>>>>>> REPLACE