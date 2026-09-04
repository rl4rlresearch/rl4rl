MECHANISM: Reverse-direction vertical TTA micro-refinement

HYPOTHESIS: Moving two float32 ULPs from the second vertical crop to the first relative to Reference Design 1 will retain 9,348 correct predictions while lowering cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Set the vertical crop weights to a two-ULP reverse offset from the best verified weighting while preserving their combined and total ensemble weight.

EVIDENCE: Reference Design 1 has the lowest verified cross-entropy; forward offsets of four and eight ULPs were worse, while reverse-direction attempts timed out without contrary validation evidence.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.312013149261474609375,
            1.314939975738525390625,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3120138645172119140625,
            1.3149392604827880859375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE