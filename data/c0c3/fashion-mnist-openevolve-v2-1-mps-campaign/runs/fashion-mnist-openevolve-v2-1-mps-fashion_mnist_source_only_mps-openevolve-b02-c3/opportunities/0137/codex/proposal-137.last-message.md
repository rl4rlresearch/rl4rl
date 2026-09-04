MECHANISM: Reverse center–vertical TTA group micro-reweighting

HYPOTHESIS: Raising the center-crop weight by one float32 ULP while lowering each vertical-crop weight by one ULP will preserve all 9,348 correct predictions and reduce cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Preserve the total ensemble weight and best verified temperature while transferring the smallest exactly balanced weight increment from the vertical crop pair to the center crop.

EVIDENCE: The opposite center-to-vertical transfer did not improve cross-entropy, while within-pair crop weights and temperature have already been bracketed; this makes the untested reverse center–vertical direction the most informative remaining local refinement.

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
            3.0000002384185791015625,
            1.31201350688934326171875,
            1.31493937969207763671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
>>>>>>> REPLACE