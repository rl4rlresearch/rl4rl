MECHANISM: Bracketed directional TTA weight refinement

HYPOTHESIS: A half-step vertical-weight transfer from Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Set the vertical crop weights halfway between Reference Designs 1 and 2, preserving their combined and total ensemble weight.

EVIDENCE: Reference Design 1 is the best verified result; a full 1/1048576 transfer in Reference Design 2 slightly worsened cross-entropy, so the midpoint is the most informative local refinement.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3120269775390625,
            1.3149261474609375,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.312013149261474609375,
            1.314939975738525390625,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE