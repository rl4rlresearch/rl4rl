MECHANISM: Paired-view consistency regularization

HYPOTHESIS: Adding symmetric flip-consistency regularization to Reference Design 3’s qualified paired-view training will exceed 9,315 correct predictions by stabilizing the orientation-invariant decision boundary.

INTENDED_EDIT: Adopt probability-averaged translated/flip inference at temperature 0.78, jointly supervise horizontal view pairs, and penalize disagreement between their class distributions.

EVIDENCE: Paired horizontal supervision raised correctness from 9,274 to 9,315, while temperature 0.78 preserved those predictions and achieved the best cross-entropy; explicit consistency directly strengthens the invariance responsible for that gain.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        logits = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        for crop, weight in zip(crops, crop_weights):
            logits = logits + weight * self._forward_once(crop)
            logits = logits + weight * self._forward_once(crop.flip(-1))
        return logits / (2.0 * sum(crop_weights))
=======
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.78
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
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
>>>>>>> REPLACE