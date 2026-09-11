MECHANISM: Hybrid logit–probability test-time aggregation

HYPOTHESIS: Blending 25% reliability-weighted probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by reducing the influence of extreme logits from unreliable transformed views while preserving the ensemble’s established center and flip-consistency weighting.

INTENDED_EDIT: Keep training and compute unchanged; during evaluation, combine the existing weighted logit pool with a conservative weighted arithmetic probability pool before prediction and calibration.

EVIDENCE: The current weighted ten-view logit ensemble achieved the best result of 9,325 correct, while center-anchored training fell to 9,324; this motivates refining inference aggregation without risking the repeated training-time failures caused by more elaborate training changes.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
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
        return calibration.unsqueeze(1) * pooled_logits
=======
        offset_logits = view_logits.mean(dim=1)
        offset_probabilities = F.softmax(view_logits, dim=-1).mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
        weight_sum = offset_weights.sum(dim=0).unsqueeze(1)
        logit_pool = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / weight_sum
        probability_pool = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / weight_sum
        pooled_logits = (
            0.75 * logit_pool
            + 0.25 * probability_pool.clamp_min(1.0e-8).log()
        )
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
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE