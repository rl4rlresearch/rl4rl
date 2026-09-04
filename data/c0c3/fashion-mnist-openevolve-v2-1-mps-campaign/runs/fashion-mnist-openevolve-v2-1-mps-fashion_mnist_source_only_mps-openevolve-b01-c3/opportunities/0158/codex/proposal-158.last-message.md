MECHANISM: Agreement-curved confidence calibration

HYPOTHESIS: Restoring the verified 10% flip-inconsistency penalty and mildly reducing confidence for intermediate-agreement ensembles will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.

INTENDED_EDIT: Use the accuracy-preserving 10% pair penalty and add a small quadratic uncertainty correction to validation-logit calibration without changing predicted classes.

EVIDENCE: The 10% penalty achieved 9,289 correct with the best reported cross-entropy, while every tested lower penalty lost one prediction and stronger/adaptive suppression did not improve the objective; this motivates preserving its decisions and refining only confidence.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.875 + 0.125 * flip_consistency)
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