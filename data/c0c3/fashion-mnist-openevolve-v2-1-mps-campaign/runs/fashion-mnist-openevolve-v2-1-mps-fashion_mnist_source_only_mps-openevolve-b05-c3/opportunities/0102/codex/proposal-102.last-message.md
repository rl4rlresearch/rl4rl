MECHANISM: Decision-preserving TTA temperature sharpening

HYPOTHESIS: Scaling the ensemble log-probabilities by 1.05 will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.235889 to approximately 0.230772.

INTENDED_EDIT: Apply the verified 1.05× evaluation-only sharpening factor without changing training or predicted classes.

EVIDENCE: Reference Design 3 used this exact change and retained 9,206 correct predictions while improving validation_score from 9206.404567 to 9206.406249.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE