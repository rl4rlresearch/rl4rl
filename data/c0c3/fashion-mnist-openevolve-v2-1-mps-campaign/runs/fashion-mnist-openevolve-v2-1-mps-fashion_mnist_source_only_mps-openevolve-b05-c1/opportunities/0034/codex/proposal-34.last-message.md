MECHANISM: Conservative post-fusion temperature refinement

HYPOTHESIS: Scaling the fused log-probabilities by 1.15 will preserve all 9,249 class decisions while reducing validation cross-entropy below 0.215002.

INTENDED_EDIT: Increase probability-ensemble sharpening from 1.1 to 1.15 without changing training, parameters, inference views, or predicted argmaxes.

EVIDENCE: Increasing the fusion scale from 1.0 to 1.1 preserved 9,249 correct predictions and reduced cross-entropy from 0.216368 to 0.215002; the prior 1.15 attempt timed out without subject-level contrary evidence.

<<<<<<< SEARCH
        return 1.1 * probabilities.clamp_min(1e-8).log()
=======
        return 1.15 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE