MECHANISM: Flip-paired consistency training

HYPOTHESIS: Training every presented image alongside its horizontal mirror with a small symmetric consistency loss will reduce the flip instability that inference-time penalties identified, exceeding 9,289 correct predictions while retaining the best validated inference pooling and calibration.

INTENDED_EDIT: Restore the verified uniform 10% inconsistency penalty and agreement-curved calibration, then jointly train original/flipped pairs using smoothed classification loss plus symmetric KL consistency.

EVIDENCE: Uniform 10% pair suppression reached 9,289 correct, whereas adaptive or member-specific suppression did not improve correctness; this suggests reducing flip disagreement during learning is more promising than further inference-time reweighting.

<<<<<<< SEARCH
        view_probabilities = view_logits.softmax(dim=-1)
        flip_disagreement = 0.5 * (
            view_probabilities[:, 0] - view_probabilities[:, 1]
        ).abs().sum(dim=-1)
        inconsistency_penalty = (
            0.10 + 0.15 * flip_disagreement
        ) * (1.0 - flip_consistency)
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (1.0 - inconsistency_penalty)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
=======
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.22775 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_logits = model(torch.cat((images, images.flip(-1)), dim=0))
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    flipped_log_probabilities = F.log_softmax(flipped_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            flipped_log_probabilities.exp(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probabilities,
            original_log_probabilities.exp(),
            reduction="batchmean",
        )
    )
    return classification_loss + 0.05 * consistency_loss
>>>>>>> REPLACE