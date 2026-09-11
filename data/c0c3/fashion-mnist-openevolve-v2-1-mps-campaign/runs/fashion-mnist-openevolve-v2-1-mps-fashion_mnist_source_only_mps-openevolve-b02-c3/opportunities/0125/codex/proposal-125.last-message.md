MECHANISM: One-ULP reverse vertical TTA refinement

HYPOTHESIS: Moving one float32 ULP from the second vertical crop to the first will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Apply the smallest representable reverse-direction adjustment to the current best vertical crop weights while preserving their combined weight.

EVIDENCE: The current weighting has the lowest verified cross-entropy; smaller first-crop weights in both reference designs were worse, while previous reverse-direction probes timed out and supplied no contrary validation evidence.

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
            1.31201374530792236328125,
            1.31493937969207763671875,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE