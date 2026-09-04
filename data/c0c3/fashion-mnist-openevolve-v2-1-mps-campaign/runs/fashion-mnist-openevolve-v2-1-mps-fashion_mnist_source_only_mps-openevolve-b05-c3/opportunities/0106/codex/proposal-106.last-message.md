MECHANISM: Decision-preserving TTA temperature sharpening

HYPOTHESIS: Scaling the qualified ensemble log-probabilities by 1.05 will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.235889 to approximately 0.230772.

INTENDED_EDIT: Multiply only the evaluation-time ensemble log-probabilities by 1.05, leaving training and predicted classes unchanged.

EVIDENCE: Reference Design 3 verified this exact change at 9,206 correct and 0.230772 cross-entropy, outperforming the current design’s tied correct count and 0.235889 cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE