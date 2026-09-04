MECHANISM: Quadratically interpolated agreement-conditioned calibration

HYPOTHESIS: A 7.708% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.

INTENDED_EDIT: Restore the center-view weight used by the best reference and apply a weaker, quadratically estimated agreement-conditioned logit scale.

EVIDENCE: The 0%, 15%, and 22.5% adjustments produced cross-entropies of 0.20670592765808105, 0.20670405883789061, and 0.20675360946655275; quadratic interpolation estimates the minimum near 7.708%, while positive scaling preserves the pooled argmax.

<<<<<<< SEARCH
        pooled_logits = (
            1.5578643 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578643
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
            0.92292 + 0.07708 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE