MECHANISM: View-agreement-conditioned logit calibration

HYPOTHESIS: Downscaling logits only for examples with disagreeing TTA views will retain all 9,287 predictions while reducing validation cross-entropy below 0.2067059268951416.

INTENDED_EDIT: Preserve the verified logit pooling and apply a positive per-image temperature based on agreement between all ten view predictions.

EVIDENCE: Probability pooling lost one correct prediction, while center-weight refinement reached an accuracy boundary; positive scalar calibration uses view disagreement without changing the pooled argmax.

<<<<<<< SEARCH
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        return 1.22775 * pooled_logits
=======
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (0.85 + 0.15 * view_agreement)
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE