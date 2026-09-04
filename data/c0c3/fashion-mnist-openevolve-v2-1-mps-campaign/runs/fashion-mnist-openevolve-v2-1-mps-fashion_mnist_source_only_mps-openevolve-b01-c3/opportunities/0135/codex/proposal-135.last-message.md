MECHANISM: Local quadratic refinement of agreement-conditioned calibration

HYPOTHESIS: A 7.788524% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935928344725.

INTENDED_EDIT: Restore the best verified pooling configuration and move its agreement-conditioning strength from 7.727% to the local quadratic minimum.

EVIDENCE: The verified 7.708%, 7.727%, and 15% adjustments produced cross-entropies of 0.20668936004638672, 0.20668935928344725, and 0.20670405883789061; quadratic interpolation estimates a minimum near 7.788524%, while positive scaling cannot change pooled argmax predictions.

<<<<<<< SEARCH
        pooled_logits = (
            1.5578756246377452 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578756246377452
        return 1.22775 * pooled_logits
=======
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
>>>>>>> REPLACE