MECHANISM: Stronger decision-preserving TTA temperature sharpening

HYPOTHESIS: Scaling ensemble log-probabilities by 1.10 will preserve exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.

INTENDED_EDIT: Increase only the evaluation-time sharpening factor from 1.05 to 1.10.

EVIDENCE: The verified 1.05× scale preserved 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772; the prior 1.10× attempt timed out and provided no performance evidence against the stronger calibration.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE