MECHANISM: Stronger evaluation-only temperature sharpening

HYPOTHESIS: Restoring constant 0.02 label smoothing will recover 9,328 correct predictions, while sharpening fused validation logits from temperature 0.90 to 0.85 will preserve those argmax predictions and reduce cross-entropy below 0.194148.

INTENDED_EDIT: Restore the best-count constant smoothing regime and apply temperature 0.85 to the existing confidence-adaptive flip ensemble.

EVIDENCE: Constant 0.02 smoothing produced 9,328 correct, and temperature 0.90 preserved all 9,328 while lowering cross-entropy from 0.201899 to 0.194148; a modest additional sharpening directly tests whether calibration can improve further without affecting predicted classes.

<<<<<<< SEARCH
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
=======
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.85
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE