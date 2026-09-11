MECHANISM: Argmax-invariant ensemble temperature calibration

HYPOTHESIS: Dividing the qualified top-four model’s probability-mixture logits by 1.05 will preserve its 9,322 correct predictions while lowering validation cross-entropy below 0.192551.

INTENDED_EDIT: Apply temperature 1.05 only to evaluation logits, leaving initialization, training, parameters, and predictions by argmax unchanged.

EVIDENCE: The current top-four design has the best verified count, and arithmetic probability averaging previously improved ensemble cross-entropy; calibration remains untested because prior attempts timed out rather than producing contrary validation evidence.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        return (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        ) / 1.05
>>>>>>> REPLACE