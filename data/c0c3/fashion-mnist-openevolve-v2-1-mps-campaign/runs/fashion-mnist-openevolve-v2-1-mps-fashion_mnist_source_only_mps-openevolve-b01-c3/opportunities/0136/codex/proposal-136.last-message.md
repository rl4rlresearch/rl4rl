MECHANISM: Local quadratic refinement of agreement-conditioned calibration

HYPOTHESIS: A 7.7757513% disagreement adjustment will preserve all 9,287 pooled predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935852050782.

INTENDED_EDIT: Restore the best verified center-view pooling weight and apply the quadratic-minimum agreement-conditioned logit calibration.

EVIDENCE: The verified 7.708%, 7.727%, and 7.788524% adjustments produced cross-entropies of 0.20668936004638672, 0.20668935928344725, and 0.20668935852050782; quadratic interpolation places the next minimum near 7.7757513%, while positive scaling cannot change argmax predictions.

<<<<<<< SEARCH
        pooled_logits = (
            1.557939852073332 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557939852073332
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
            0.922242487 + 0.077757513 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE