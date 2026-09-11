MECHANISM: Confidence-bounded probability-space test-time ensembling

HYPOTHESIS: Averaging bounded per-view probabilities instead of logits will exceed 9,287 correct predictions by reducing the influence of confidently wrong transformed views on disagreement cases.

INTENDED_EDIT: Keep training and all learned parameters unchanged; replace logit-space test-time pooling with center-weighted probability pooling, then convert the ensemble back to logits and retain the best verified agreement calibration.

EVIDENCE: Agreement-conditioned calibration improved cross-entropy while preserving 9,287 correct, identifying view disagreement as useful reliability information; spatial and channel-pooling changes reduced accuracy, motivating a targeted inference-only aggregation change.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
=======
        view_probabilities = view_logits.softmax(dim=-1)
        offset_probabilities = view_probabilities.mean(dim=1)
        pooled_probabilities = (
            1.5578022218542173 * offset_probabilities[0]
            + offset_probabilities[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_logits = pooled_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_probabilities.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE