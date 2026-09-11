MECHANISM: Train–evaluation translation-distribution alignment

HYPOTHESIS: Matching stochastic training translations to the validated anisotropic TTA weights will exceed 9,284 correct predictions by training the model for the same directional robustness emphasized during evaluation.

INTENDED_EDIT: Replace the symmetric training translation probabilities with the current evaluation ensemble’s normalized center, vertical, horizontal, diagonal, and radius-two weights.

EVIDENCE: Calibration is saturated at 9,284 correct, while the best evaluation ensemble assigns substantially different weights to vertical and horizontal one-pixel shifts; training still samples those directions equally, leaving a concrete train–evaluation mismatch.

<<<<<<< SEARCH
    translation_weights = images.new_tensor(
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
    )
=======
    translation_weights = images.new_tensor(
        (
            0.003125, 0.006250, 0.009375, 0.006250, 0.003125,
            0.006250, 0.045000, 0.1084375, 0.045000, 0.006250,
            0.009375, 0.06953125, 0.3640625, 0.06953125, 0.009375,
            0.006250, 0.045000, 0.1084375, 0.045000, 0.006250,
            0.003125, 0.006250, 0.009375, 0.006250, 0.003125,
        )
    )
>>>>>>> REPLACE