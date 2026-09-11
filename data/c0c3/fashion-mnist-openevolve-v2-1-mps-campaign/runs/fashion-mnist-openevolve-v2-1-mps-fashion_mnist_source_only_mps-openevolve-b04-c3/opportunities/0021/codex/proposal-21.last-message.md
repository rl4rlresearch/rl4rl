MECHANISM: Fine-grained upper-side cosine-floor refinement

HYPOTHESIS: A 12.5% terminal learning-rate floor will exceed 9,233 correct predictions by preserving the demonstrated benefit of late optimization while remaining well below the harmful 20% floor.

INTENDED_EDIT: Raise only the cosine schedule’s terminal learning-rate floor from 10% to 12.5% of peak, leaving architecture, augmentation, optimizer, batch size, and inference unchanged.

EVIDENCE: The 10% floor achieved 9,233 correct with better cross-entropy than the tied 7.5% floor, while the 20% floor fell to 9,226; 12.5% is the closest untested refinement in the direction favored by that evidence.

<<<<<<< SEARCH
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE