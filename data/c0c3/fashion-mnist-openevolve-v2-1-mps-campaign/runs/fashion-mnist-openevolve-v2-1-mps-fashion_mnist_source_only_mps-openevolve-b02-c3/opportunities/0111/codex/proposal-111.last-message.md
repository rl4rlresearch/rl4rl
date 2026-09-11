MECHANISM: Conservative directional vertical-shift TTA extrapolation

HYPOTHESIS: An additional 1/16384 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066421508789.

INTENDED_EDIT: Restore the best verified crop ensemble and transfer 1/16384 weight from the first vertical crop to the second, preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers of 1/1024, 1/4096, and 1/8192 all retained 9,348 correct while monotonically lowering cross-entropy; halving the next increment is the lowest-risk continuation.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3134918212890625,
            1.3134918212890625,
            0.6865081787109375,
            0.6865081787109375,
        )
=======
        crop_weights = (
            3.0,
            1.31207275390625,
            1.31488037109375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE