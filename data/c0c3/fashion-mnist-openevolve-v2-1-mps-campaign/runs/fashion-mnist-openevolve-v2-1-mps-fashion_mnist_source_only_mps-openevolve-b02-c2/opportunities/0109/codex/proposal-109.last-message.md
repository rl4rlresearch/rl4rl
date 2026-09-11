MECHANISM: Validation-only temperature calibration

HYPOTHESIS: Applying temperature 1.05 to the flip-mixture logits will preserve the current design’s 9,322 correct predictions while reducing validation cross-entropy below 0.192551.

INTENDED_EDIT: Mildly soften the existing arithmetic probability-mixture output without changing training, parameters, runtime materially, or validation argmaxes.

EVIDENCE: Exact top-four saliency produced the best verified count of 9,322; positive temperature scaling preserves those predictions and cleanly isolates the calibration idea from the timed-out experiment that combined it with altered tail training.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        return (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        ) / 1.05
>>>>>>> REPLACE