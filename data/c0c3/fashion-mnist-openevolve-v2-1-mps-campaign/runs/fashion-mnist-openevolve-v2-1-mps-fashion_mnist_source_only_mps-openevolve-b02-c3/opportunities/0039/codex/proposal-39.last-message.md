MECHANISM: Qualified paired-view supervision with locally optimized posterior calibration

HYPOTHESIS: Removing the harmful consistency penalty will restore 9,315 correct predictions, while temperature 0.75317 will preserve those argmax predictions and slightly reduce cross-entropy below 0.1913572048.

INTENDED_EDIT: Restore pure paired-view cross-entropy and refine the inference temperature from 0.78 to the quadratic-interpolated optimum of 0.75317.

EVIDENCE: Explicit flip consistency reduced correctness from 9,315 to 9,311; the otherwise matched paired-view design achieved 9,315, and temperatures 0.753 and 0.7532 gave successively lower cross-entropies with identical predictions, locating the minimum near 0.75317.

<<<<<<< SEARCH
        ).log() / 0.78
=======
        ).log() / 0.75317
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    first_logits, second_logits = paired_logits.chunk(2, dim=0)

    supervised_loss = F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )
    first_log_probabilities = F.log_softmax(first_logits, dim=1)
    second_log_probabilities = F.log_softmax(second_logits, dim=1)
    consistency_loss = 0.5 * (
        F.kl_div(
            first_log_probabilities,
            second_log_probabilities.exp().detach(),
            reduction="batchmean",
        )
        + F.kl_div(
            second_log_probabilities,
            first_log_probabilities.exp().detach(),
            reduction="batchmean",
        )
    )
    return supervised_loss + 0.05 * consistency_loss
=======
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE