MECHANISM: Directional vertical-shift TTA micro-weight extrapolation

HYPOTHESIS: An additional 1/1048576 vertical-weight transfer beyond Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Restore the best verified crop ensemble, then transfer 1/1048576 weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive halved directional transfers through 1/524288 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 1 is the best verified result, so another halved increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.31201934814453125,
            1.31493377685546875,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.31201267242431640625,
            1.31494045257568359375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE