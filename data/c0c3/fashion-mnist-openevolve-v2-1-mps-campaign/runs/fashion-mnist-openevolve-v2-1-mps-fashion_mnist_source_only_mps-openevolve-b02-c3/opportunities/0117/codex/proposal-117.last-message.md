MECHANISM: Directional vertical-shift TTA half-step extrapolation

HYPOTHESIS: An additional 1/524288 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656623840332.

INTENDED_EDIT: Transfer 1/524288 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers through 1/262144 retained 9,348 correct and monotonically lowered cross-entropy; the current design is the best verified result, so halving the latest successful increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.312015533447265625,
            1.314937591552734375,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE