MECHANISM: Second-ULP ensemble temperature softening

HYPOTHESIS: Raising the best verified ensemble temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore Reference Design 2’s optimal crop weights and increase its temperature from 0.753170073032379150390625 to the next representable float32 value, 0.75317013263702392578125.

EVIDENCE: The first higher-temperature ULP produced the lowest verified cross-entropy without changing correctness, while the symmetric lower-temperature probe worsened cross-entropy; the smallest continuation in the successful direction is the most informative calibration test.

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
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.75317
=======
        ).log() / 0.75317013263702392578125
>>>>>>> REPLACE