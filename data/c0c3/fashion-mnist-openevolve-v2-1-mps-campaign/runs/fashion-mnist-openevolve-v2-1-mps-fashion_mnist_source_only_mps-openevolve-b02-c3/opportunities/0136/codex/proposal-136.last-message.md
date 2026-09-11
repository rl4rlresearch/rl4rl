MECHANISM: Reverse center–horizontal TTA group reweighting

HYPOTHESIS: Restoring Reference Design 2’s best temperature and transferring one float32 ULP of total weight from the horizontal crops to the center crop will preserve 9,348 correct predictions while lowering cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Raise the center weight by one ULP, lower each horizontal weight by two ULPs to preserve total weight, and restore the best verified temperature.

EVIDENCE: The opposite center-to-horizontal transfer worsened cross-entropy without changing correctness; probing the reverse direction around Reference Design 2 is the most informative remaining group-weight test.

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
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523258686065673828125,
            0.686523377895355224609375,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.753169953823089599609375
=======
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE