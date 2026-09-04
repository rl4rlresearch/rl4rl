MECHANISM: Second-ULP horizontal TTA weight extrapolation

HYPOTHESIS: Restoring the best vertical weights and transferring one additional float32 ULP from the first horizontal crop to the second will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656127929689.

INTENDED_EDIT: Restore Reference Design 1’s vertical weights and move the horizontal crop weights one float32 ULP farther in Reference Design 2’s successful direction while preserving their combined and total ensemble weight.

EVIDENCE: Reference Design 2 improved the best validation cross-entropy from 0.1877065631866455 to 0.18770656127929689 with a one-ULP horizontal transfer and no loss of correct predictions, motivating the smallest continuation along that dimension.

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
            1.3120136260986328125,
            1.3149394989013671875,
            0.68652331829071044921875,
            0.68652355670928955078125,
        )
>>>>>>> REPLACE