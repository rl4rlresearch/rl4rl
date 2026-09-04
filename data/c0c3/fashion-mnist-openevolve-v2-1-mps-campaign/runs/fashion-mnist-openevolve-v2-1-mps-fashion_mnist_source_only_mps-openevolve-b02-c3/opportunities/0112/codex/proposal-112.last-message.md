MECHANISM: Directional vertical-shift TTA micro-extrapolation

HYPOTHESIS: An additional 1/32768 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066032409668.

INTENDED_EDIT: Restore the best verified crop ensemble and transfer 1/32768 weight from the first vertical crop to the second, preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers of 1/1024, 1/4096, 1/8192, and 1/16384 retained 9,348 correct while monotonically lowering cross-entropy; halving the latest increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.312042236328125,
            1.314910888671875,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE