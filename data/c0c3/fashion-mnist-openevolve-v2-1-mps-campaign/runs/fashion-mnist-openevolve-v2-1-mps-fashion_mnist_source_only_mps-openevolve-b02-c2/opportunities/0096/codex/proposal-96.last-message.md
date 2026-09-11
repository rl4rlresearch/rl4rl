MECHANISM: Mild label-smoothed paired supervision with arithmetic flip ensembling

HYPOTHESIS: Applying 0.02 label smoothing to the reliable hard-maximum model will exceed 9,322 correct predictions, while arithmetic probability averaging will reduce cross-entropy relative to mean-logit ensembling.

INTENDED_EDIT: Add low-overhead label smoothing to paired-view cross-entropy and restore arithmetic probability averaging for validation flips.

EVIDENCE: Hard-maximum training reliably reached 9,320 correct while repeated attention alternatives regressed or timed out; arithmetic probability ensembling retained 9,320 correct and reduced cross-entropy from 0.1926495 to 0.1922617, motivating an orthogonal calibration-oriented training regularizer.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE