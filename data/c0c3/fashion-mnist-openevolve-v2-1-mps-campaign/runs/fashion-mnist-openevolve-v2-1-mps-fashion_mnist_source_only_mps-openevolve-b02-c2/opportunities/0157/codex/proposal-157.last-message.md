MECHANISM: Late cosine decay of label smoothing

HYPOTHESIS: Holding 0.02 smoothing through the first half of training and annealing it to zero will exceed 9,328 correct predictions by retaining the verified regularization benefit while allowing late hard-target boundary refinement; if correct counts tie, it will lower cross-entropy.

INTENDED_EDIT: Replace constant 0.02 label smoothing with a schedule that remains at 0.02 for half the optimizer trajectory, then cosine-decays to zero.

EVIDENCE: Fixed 0.02 smoothing improved the reliable hard-max design from 9,320 to 9,328 correct, but increased cross-entropy from about 0.1922 to 0.2019, motivating preservation of early smoothing with less late confidence suppression.

<<<<<<< SEARCH
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
=======
    progress = min(step / max(total_steps, 1), 1.0)
    decay_progress = max(2.0 * progress - 1.0, 0.0)
    label_smoothing = 0.01 * (1.0 + math.cos(math.pi * decay_progress))
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE