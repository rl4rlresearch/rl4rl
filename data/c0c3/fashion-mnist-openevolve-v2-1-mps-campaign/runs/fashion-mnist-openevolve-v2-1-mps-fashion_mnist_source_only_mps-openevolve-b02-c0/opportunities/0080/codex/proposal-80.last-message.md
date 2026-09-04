MECHANISM: Convex post-ensemble temperature calibration

HYPOTHESIS: Scaling the verified ensemble scores from 1.275 to 1.325 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; architecture, training, and ensemble decisions remain unchanged.

EVIDENCE: Successive scales from 1.10 through 1.275 preserved all 9,253 predictions while monotonically reducing cross-entropy, and the observed marginal improvements suggest the convex calibration optimum lies near 1.325; prior higher-scale timeouts provide no contrary score evidence.

<<<<<<< SEARCH
        return 1.275 * ensemble_scores
=======
        return 1.325 * ensemble_scores
>>>>>>> REPLACE