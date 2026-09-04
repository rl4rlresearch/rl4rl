MECHANISM: Directional vertical-shift TTA pico-extrapolation

HYPOTHESIS: An additional 1/131072 vertical-weight transfer beyond Reference Design 2 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770657272338867.

INTENDED_EDIT: Restore the best verified crop weights, then transfer 1/131072 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers through 1/65536 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 2 is the best verified result, so halving its latest successful increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3125,
            1.314453125,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.31201934814453125,
            1.31493377685546875,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE