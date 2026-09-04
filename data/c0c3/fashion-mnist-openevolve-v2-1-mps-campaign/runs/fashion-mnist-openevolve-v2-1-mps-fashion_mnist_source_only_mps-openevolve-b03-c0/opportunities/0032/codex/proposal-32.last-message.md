MECHANISM: Shallower post-kick learning-rate warmup

HYPOTHESIS: Raising the warmup floor from 20% to 40% of peak will exceed 9,319 correct predictions by accelerating early optimization while retaining the warmup that outperformed full-rate cosine training.

INTENDED_EDIT: Keep the beneficial full-rate first update, then warm from 40% instead of 20% to the unchanged 2.0e-3 peak over 5% of training.

EVIDENCE: Warmup-consistent initialization scored 9,298 and removing warmup scored 9,275, versus 9,319 for the current full-rate kick followed by warmup; this motivates preserving both features while testing a less severe post-kick learning-rate drop.

<<<<<<< SEARCH
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
=======
    if progress < 0.05:
        multiplier = 0.4 + 0.6 * progress / 0.05
>>>>>>> REPLACE