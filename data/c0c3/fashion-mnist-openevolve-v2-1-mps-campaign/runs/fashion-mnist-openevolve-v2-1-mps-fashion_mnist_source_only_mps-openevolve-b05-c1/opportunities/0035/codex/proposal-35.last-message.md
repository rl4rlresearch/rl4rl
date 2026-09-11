MECHANISM: Quadratic-interpolated ensemble calibration

HYPOTHESIS: Scaling fused log-probabilities by 1.106 will preserve all 9,249 argmax predictions while reducing validation cross-entropy below 0.215002.

INTENDED_EDIT: Refine post-fusion sharpening from 1.1 to 1.106 without changing training, inference views, or parameters.

EVIDENCE: Scales 1.0, 1.1, and 1.15 produced cross-entropies 0.216368, 0.215002, and 0.215228 with identical correctness; quadratic interpolation places the minimum near 1.106.

<<<<<<< SEARCH
        return 1.1 * probabilities.clamp_min(1e-8).log()
=======
        return 1.106 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE