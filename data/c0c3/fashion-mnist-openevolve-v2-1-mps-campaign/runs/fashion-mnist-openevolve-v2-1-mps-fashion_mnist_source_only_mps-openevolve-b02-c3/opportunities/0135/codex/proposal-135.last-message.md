MECHANISM: Center-to-horizontal TTA group micro-reweighting

HYPOTHESIS: Transferring one float32 ULP of center-crop weight equally to the horizontal crops will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore Reference Design 1’s best crop weights and temperature, then lower the center weight by one ULP and raise each horizontal weight by two ULPs, preserving the exact total ensemble weight.

EVIDENCE: Reference Design 1 has the lowest verified cross-entropy; temperature, within-pair horizontal, and vertical weighting have been bracketed, while center-to-vertical transfer did not improve, making the untested center-versus-horizontal group direction the most informative orthogonal probe.

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
            2.9999997615814208984375,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523497104644775390625,
            0.686523616313934326171875,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.75317
=======
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE