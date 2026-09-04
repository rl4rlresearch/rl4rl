MECHANISM: Temperature-sharpened probability fusion

HYPOTHESIS: Scaling the fused log-probabilities by 1.1 will retain exactly 9,249 correct predictions while lowering validation cross-entropy below 0.216368.

INTENDED_EDIT: Apply mild temperature sharpening after the verified probability-space test-time augmentation, without changing its class decisions, training, or runtime materially.

EVIDENCE: Probability averaging improved correctness from 9,240 to 9,249 but worsened cross-entropy from 0.214599 to 0.216368; positive logit scaling preserves every argmax while testing whether the arithmetic ensemble is slightly underconfident.

<<<<<<< SEARCH
        return probabilities.clamp_min(1e-8).log()
=======
        return 1.1 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE