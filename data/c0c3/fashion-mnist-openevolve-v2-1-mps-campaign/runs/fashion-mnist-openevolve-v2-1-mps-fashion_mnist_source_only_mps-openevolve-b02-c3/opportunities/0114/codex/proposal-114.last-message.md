MECHANISM: Directional vertical-shift TTA nano-extrapolation

HYPOTHESIS: An additional 1/65536 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770658378601074.

INTENDED_EDIT: Restore the best verified crop weights, then transfer 1/65536 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers through the latest 1/32768 increment retained 9,348 correct and monotonically reduced cross-entropy; halving that latest successful increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3121337890625,
            1.3148193359375,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3120269775390625,
            1.3149261474609375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE