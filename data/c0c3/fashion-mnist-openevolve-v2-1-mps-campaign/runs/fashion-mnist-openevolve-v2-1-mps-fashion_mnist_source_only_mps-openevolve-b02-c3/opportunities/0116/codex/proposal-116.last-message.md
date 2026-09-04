MECHANISM: Directional vertical-shift TTA femto-extrapolation

HYPOTHESIS: An additional 1/262144 vertical-weight transfer beyond Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656814575196.

INTENDED_EDIT: Restore the best verified crop ensemble, then transfer 1/262144 weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers through 1/131072 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 1 is the best verified result, making a halved continuation the lowest-risk next probe.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.312015533447265625,
            1.314937591552734375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE