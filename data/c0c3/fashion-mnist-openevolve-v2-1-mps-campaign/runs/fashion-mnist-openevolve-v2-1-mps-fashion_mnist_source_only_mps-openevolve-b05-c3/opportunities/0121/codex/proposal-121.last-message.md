MECHANISM: Margin-adaptive decision-preserving calibration

HYPOTHESIS: Scaling ensemble log-probabilities by 1.10–1.20 according to the top-two probability margin will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.

INTENDED_EDIT: Replace fixed 1.10× outer sharpening with stronger sharpening for high-agreement predictions and retain 1.10× for ambiguous predictions.

EVIDENCE: Increasing fixed outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and lowered cross-entropy from 0.227083 to 0.224459; margin conditioning tests further sharpening while limiting its effect on error-prone ambiguous examples.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        probabilities = (probability_sum / 10.0).clamp_min(1e-8)
        top_two = probabilities.topk(2, dim=1).values
        confidence_margin = top_two[:, 0] - top_two[:, 1]
        outer_scale = 1.10 + 0.10 * confidence_margin
        return probabilities.log().mul_(outer_scale.unsqueeze(1))
>>>>>>> REPLACE