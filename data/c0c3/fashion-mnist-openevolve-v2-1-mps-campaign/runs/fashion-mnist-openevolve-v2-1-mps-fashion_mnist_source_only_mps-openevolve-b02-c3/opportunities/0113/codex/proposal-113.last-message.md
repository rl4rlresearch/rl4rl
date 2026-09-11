MECHANISM: Directional vertical-shift TTA micro-extrapolation

HYPOTHESIS: An additional 1/32768 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066032409668.

INTENDED_EDIT: Transfer 1/32768 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.

EVIDENCE: Successive directional transfers through 1/16384 retained 9,348 correct and monotonically lowered cross-entropy; the prior 1/32768 attempt timed out and therefore supplied no contrary validation evidence.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.31207275390625,
            1.31488037109375,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.312042236328125,
            1.314910888671875,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE