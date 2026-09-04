MECHANISM: Moderately elevated cosine learning-rate floor

HYPOTHESIS: Raising the terminal learning-rate multiplier from 10% to 11.25% will exceed 9,328 correct predictions by preserving slightly stronger late hard-label updates.

INTENDED_EDIT: Increase the cosine schedule’s terminal learning rate from 2.10e-4 to 2.3625e-4 while preserving its initial learning rate and all other behavior.

EVIDENCE: Lowering the floor to 2% reduced validation correct from 9,328 to 9,298; the prior 11.25% verification timed out and therefore supplied no contrary accuracy evidence.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.1125 + 0.8875 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE