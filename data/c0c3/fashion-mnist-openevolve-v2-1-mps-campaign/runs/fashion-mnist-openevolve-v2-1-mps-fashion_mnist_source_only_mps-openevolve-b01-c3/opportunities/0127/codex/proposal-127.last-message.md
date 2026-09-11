MECHANISM: Stronger view-agreement-conditioned logit calibration

HYPOTHESIS: Doubling the disagreement adjustment to 0.30 will retain exactly 9,287 correct predictions because it applies only a positive per-image logit scale, while reducing validation cross-entropy below 0.20670405883789061.

INTENDED_EDIT: Restore the best verified center-view weight and strengthen the verified agreement-conditioned calibration from 15% to 30%.

EVIDENCE: Reference Design 1 retained 9,287 correct and improved cross-entropy from 0.20670592765808105 to 0.20670405883789061 with a 15% agreement adjustment; a doubled adjustment is an informative accuracy-invariant line-search step.

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
        calibration = 1.22775 * (0.70 + 0.30 * view_agreement)
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE