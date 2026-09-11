MECHANISM: Center-to-vertical TTA group micro-reweighting

HYPOTHESIS: Transferring one float32 ULP of center-crop weight equally to the two vertical crops while restoring the best verified temperature will preserve all 9,348 correct predictions and reduce validation cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore Reference Design 2’s temperature, lower the center weight by one float32 ULP, and raise each vertical weight by one float32 ULP, preserving total ensemble weight and the established horizontal weighting.

EVIDENCE: Reference Design 2 is the best verified calibration; adjacent temperature and relative horizontal/vertical probes were worse, leaving center-versus-shifted group weighting as an untested orthogonal refinement. The vertical pair is the more heavily weighted shifted group, motivating the smallest exact transfer toward it.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
=======
        crop_weights = (
            2.9999997615814208984375,
            1.31201374530792236328125,
            1.31493961811065673828125,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.75317013263702392578125
=======
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE