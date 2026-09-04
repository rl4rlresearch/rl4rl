MECHANISM: Conservative terminal label-smoothing taper

HYPOTHESIS: Holding the verified 0.02 smoothing for 75% of training and tapering only to 0.01 will retain or exceed 9,328 correct predictions while improving cross-entropy over constant smoothing.

INTENDED_EDIT: Use confidence-adaptive probability fusion and cosine-taper label smoothing from 0.02 to 0.01 during the final quarter.

EVIDENCE: Constant 0.02 smoothing achieved 9,328 correct, while an earlier, complete decay to zero retained 9,325 and substantially lowered cross-entropy; a later partial taper targets that calibration benefit without discarding most of the accuracy-producing regularization.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        top_two = log_probabilities.topk(2, dim=2).values
        margins = top_two[..., 0] - top_two[..., 1]
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    taper_progress = min(max((progress - 0.75) / 0.25, 0.0), 1.0)
    label_smoothing = 0.01 + 0.01 * 0.5 * (
        1.0 + math.cos(math.pi * taper_progress)
    )
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