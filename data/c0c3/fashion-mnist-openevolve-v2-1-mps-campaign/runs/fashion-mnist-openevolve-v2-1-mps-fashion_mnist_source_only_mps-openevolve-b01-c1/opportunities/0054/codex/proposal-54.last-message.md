MECHANISM: Sharpened cosine flip-pair curriculum

HYPOTHESIS: A steeper 25%-to-50% flip-pair curriculum will exceed 9,247 correct predictions by spending more of early training on individual-view features and concentrating validation-aligned pair supervision later, while preserving the successful endpoints and mean allocation.

INTENDED_EDIT: Sharpen the existing cosine ramp with a symmetric nonlinear transform, leaving architecture, compute, optimizer, augmentation, and all other losses unchanged.

EVIDENCE: The 25%-to-50% cosine ramp improved correctness from 9,246 to 9,247, whereas fixed 43.75% supervision regressed to 9,242; this suggests stronger pair supervision is useful when delayed rather than applied uniformly.

<<<<<<< SEARCH
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
=======
    pair_phase = 0.5 - 0.5 * math.cos(math.pi * progress)
    pair_phase_sq = pair_phase * pair_phase
    inverse_phase_sq = (1.0 - pair_phase) * (1.0 - pair_phase)
    sharpened_pair_phase = pair_phase_sq / (
        pair_phase_sq + inverse_phase_sq
    )
    pair_weight = 0.25 + 0.25 * sharpened_pair_phase
>>>>>>> REPLACE