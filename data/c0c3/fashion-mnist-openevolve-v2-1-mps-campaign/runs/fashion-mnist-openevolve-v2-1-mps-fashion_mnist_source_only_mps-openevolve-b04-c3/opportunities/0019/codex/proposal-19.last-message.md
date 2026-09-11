MECHANISM: Fine-grained nonzero-floor cosine optimization

HYPOTHESIS: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.

INTENDED_EDIT: Change the zero-terminal cosine schedule to decay from 2.5e-3 to 1.875e-4, preserving all other qualified behavior.

EVIDENCE: The 10% floor improved correctness from 9,229 to 9,233 and reduced cross-entropy, while increasing it to 20% fell to 9,226; testing 7.5% narrowly refines the supported region without changing runtime.

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE