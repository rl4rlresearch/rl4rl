MECHANISM: Separable full-grid translation marginalization

HYPOTHESIS: Adding weak, factorized diagonal crop views will exceed 9,348 correct predictions by recovering images requiring simultaneous horizontal and vertical alignment while preserving the dominant verified axial consensus.

INTENDED_EDIT: Extend evaluation TTA from five axial crops to the full 3×3 translation grid, assign each diagonal the separable product of its axial weights divided by the center weight, and restore the best verified temperature.

EVIDENCE: Hierarchical orientation-then-crop fusion reached 9,348 correct, whereas changing the learned spatial architecture reduced correctness; this motivates preserving the model and fusion while filling the remaining diagonal gap in its translation ensemble.

<<<<<<< SEARCH
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
=======
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 2:30],
        )
>>>>>>> REPLACE

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
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
            (
                1.3120136260986328125
                * 0.686523377895355224609375
                / 3.0
            ),
            (
                1.3120136260986328125
                * 0.686523497104644775390625
                / 3.0
            ),
            (
                1.3149394989013671875
                * 0.686523377895355224609375
                / 3.0
            ),
            (
                1.3149394989013671875
                * 0.686523497104644775390625
                / 3.0
            ),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ).log() / 0.753170073032379150390625
=======
        ).log() / 0.753169953823089599609375
>>>>>>> REPLACE