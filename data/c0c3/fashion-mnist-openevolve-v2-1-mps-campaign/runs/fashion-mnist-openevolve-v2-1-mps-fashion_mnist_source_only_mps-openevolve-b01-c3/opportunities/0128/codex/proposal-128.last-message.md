MECHANISM: Stronger view-agreement-conditioned logit calibration

HYPOTHESIS: A 30% disagreement adjustment will preserve exactly 9,287 correct predictions via positive scalar scaling while reducing cross-entropy below 0.20670405883789061.

INTENDED_EDIT: Restore the best verified center-view weight and double the agreement-conditioned calibration strength from 15% to 30%.

EVIDENCE: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; the prior 30% attempt only timed out, leaving this accuracy-invariant line-search step unresolved.

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
        calibration = 1.22775 * (0.70 + 0.30 * view_agreement)
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE