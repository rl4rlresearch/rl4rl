MECHANISM: Post-fusion temperature sharpening

HYPOTHESIS: Scaling the fused log-probabilities by 1.1 will preserve all 9,249 class decisions while lowering validation cross-entropy below 0.216368.

INTENDED_EDIT: Sharpen the probability-space test-time ensemble after fusion without changing training, parameters, or predicted argmaxes.

EVIDENCE: The prior 1.1-scaling attempt measured the same 9,249 correct predictions with cross-entropy reduced from 0.216368 to 0.215002; its timeout was unrelated to this computation-neutral calibration.

<<<<<<< SEARCH
        return probabilities.clamp_min(1e-8).log()
=======
        return 1.1 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE