MECHANISM: Moderately elevated cosine learning-rate floor

HYPOTHESIS: Raising the terminal learning-rate multiplier from 10% to 11.25% will exceed 9,328 correct predictions by preserving slightly stronger late hard-label updates without the larger jump of the unresolved 12.5% attempt.

INTENDED_EDIT: Increase the cosine schedule’s terminal learning rate from 2.10e-4 to 2.3625e-4 while keeping its initial rate and all other behavior unchanged.

EVIDENCE: Lowering the floor from 10% to 2% reduced validation correct from 9,328 to 9,298, while the 12.5% attempt timed out rather than producing negative accuracy evidence; a smaller upward step is the most direct test of the observed learning-rate direction.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.1125 + 0.8875 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE