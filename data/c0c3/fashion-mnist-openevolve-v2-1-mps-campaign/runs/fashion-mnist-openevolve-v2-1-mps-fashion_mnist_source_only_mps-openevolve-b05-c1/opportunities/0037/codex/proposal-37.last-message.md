MECHANISM: Local convex temperature recalibration

HYPOTHESIS: Scaling fused log-probabilities by 1.104 will preserve all 9,249 argmax predictions while reducing validation cross-entropy below 0.2150005993.

INTENDED_EDIT: Refine the inference-only probability-ensemble sharpening factor from 1.106 to 1.104.

EVIDENCE: Scale 1.1063 worsened cross-entropy to 0.2150007458 relative to 0.2150005993 at 1.106, while 1.1 was also worse at 0.2150019497; local quadratic interpolation of these three identical-correctness results places the minimum near 1.104.

<<<<<<< SEARCH
        return 1.106 * probabilities.clamp_min(1e-8).log()
=======
        return 1.104 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE