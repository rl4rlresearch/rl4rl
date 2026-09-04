MECHANISM: One-ULP reverse vertical TTA refinement

HYPOTHESIS: Moving one float32 ULP from the second vertical crop to the first relative to Reference Design 1 will retain 9,348 correct predictions while lowering cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Restore the best verified ensemble and apply the smallest representable reverse-direction vertical-weight transfer, preserving combined vertical and total weight.

EVIDENCE: Reference Design 1 has the lowest verified cross-entropy; forward offsets of four and eight ULPs were worse, while larger reverse probes timed out without contrary validation evidence.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.312042236328125,
            1.314910888671875,
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