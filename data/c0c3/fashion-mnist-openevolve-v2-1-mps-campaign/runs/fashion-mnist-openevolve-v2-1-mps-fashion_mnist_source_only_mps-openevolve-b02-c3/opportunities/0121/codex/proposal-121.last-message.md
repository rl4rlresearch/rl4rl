MECHANISM: Bracketed reverse-direction vertical TTA refinement

HYPOTHESIS: Transferring 1/2097152 weight from the second vertical crop back to the first will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Symmetrically probe the untested side of the current best crop weighting while preserving the combined vertical and total ensemble weights.

EVIDENCE: The current design is best; both the 1/1048576 forward transfer in Reference Design 1 and its half-step in Reference Design 2 worsened cross-entropy, motivating an equal-distance half-step in the reverse direction.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.312014102935791015625,
            1.314939022064208984375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE