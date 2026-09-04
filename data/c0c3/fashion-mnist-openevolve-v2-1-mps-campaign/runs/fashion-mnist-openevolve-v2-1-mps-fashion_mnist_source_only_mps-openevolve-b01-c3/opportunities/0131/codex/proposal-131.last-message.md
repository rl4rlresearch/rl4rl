MECHANISM: Midpoint agreement-conditioned logit calibration

HYPOTHESIS: A 22.5% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.

INTENDED_EDIT: Restore the best verified center-view weight and interpolate calibration strength halfway between the verified 15% adjustment and the unresolved 30% probe.

EVIDENCE: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; three 30% attempts timed out without subject-level evidence, so 22.5% is an informative intermediate line-search point.

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
        calibration = 1.22775 * (0.775 + 0.225 * view_agreement)
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE