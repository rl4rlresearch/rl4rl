MECHANISM: Separable full-grid translation fusion

HYPOTHESIS: Adding low-weight diagonal crops derived from the verified cardinal weights will exceed 9,348 correct predictions by correcting examples sensitive to simultaneous horizontal and vertical displacement.

INTENDED_EDIT: Extend inference-time augmentation from five cardinal crops to the full 3×3 translation grid, assigning each diagonal the separable product of its corresponding vertical and horizontal weights divided by the center weight.

EVIDENCE: The current weighted cardinal-crop power mean achieves the best verified 9,348 correct predictions, while changing orientation fusion did not improve correctness; diagonal translations test a new invariance axis without changing the successful model, training, or fusion rule.

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
            1.3120136260986328125
            * 0.686523377895355224609375
            / 3.0,
            1.3120136260986328125
            * 0.686523497104644775390625
            / 3.0,
            1.3149394989013671875
            * 0.686523377895355224609375
            / 3.0,
            1.3149394989013671875
            * 0.686523497104644775390625
            / 3.0,
        )
>>>>>>> REPLACE