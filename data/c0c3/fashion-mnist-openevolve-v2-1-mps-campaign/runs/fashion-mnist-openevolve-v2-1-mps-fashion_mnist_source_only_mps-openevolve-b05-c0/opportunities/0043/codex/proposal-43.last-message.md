MECHANISM: Nonzero cosine learning-rate floor

HYPOTHESIS: Keeping the learning rate at 5% of its peak near the end of training will exceed 9,254 correct predictions by sustaining useful tail optimization while EMA limits late-update noise.

INTENDED_EDIT: Change the cosine schedule from decay-to-zero to decay-to-5%-of-peak, preserving the model, peak learning rate, loss, EMA, batch size, and evaluation scaling.

EVIDENCE: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that additional finite-budget optimization helps; a small learning-rate floor targets the under-optimized tail without raising the already-proven peak rate.

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE