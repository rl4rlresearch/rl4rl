MECHANISM: Decision-preserving intermediate TTA temperature sharpening

HYPOTHESIS: Increasing evaluation-time sharpening from 1.05× to 1.075× will preserve exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.

INTENDED_EDIT: Scale the final ensemble log-probabilities by 1.075 instead of 1.05, leaving training, EMA, architecture, and argmax predictions unchanged.

EVIDENCE: Moving from unsharpened logits to 1.05× sharpening preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772; 1.075× is an informative midpoint toward the timed-out 1.10× test.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.075)
>>>>>>> REPLACE