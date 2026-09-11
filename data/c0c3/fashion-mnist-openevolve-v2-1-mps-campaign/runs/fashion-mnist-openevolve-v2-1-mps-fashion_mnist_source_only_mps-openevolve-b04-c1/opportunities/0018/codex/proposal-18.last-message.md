MECHANISM: Low-floor cosine tail convergence

HYPOTHESIS: Reducing the final learning-rate floor from 10% to 1% will exceed 9,286 correct predictions by stabilizing the late trajectory averaged by the proven 0.02 parameter-only EMA.

INTENDED_EDIT: Preserve the architecture, paired-view objective, optimizer, and EMA while changing cosine decay from a 2.1e-4 minimum learning rate to 2.1e-5.

EVIDENCE: Shortening EMA lag from rate 0.01 to 0.02 improved correct predictions from 9,282 to 9,286, but further shortening to 0.04 regressed; reducing late parameter motion tests whether the successful 0.02 EMA can average a more converged trajectory without sacrificing smoothing.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.01 + 0.99 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE