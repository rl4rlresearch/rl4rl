MECHANISM: Train–evaluation translation-distribution alignment

HYPOTHESIS: Matching radius-2 training probabilities to the best verified distance-stratified TTA weights will exceed 9,268 correct predictions while preserving the total exposure to outer translations.

INTENDED_EDIT: Reallocate the unchanged 0.10 radius-2 training probability from corner shifts to axial shifts, using the current evaluation ensemble’s 0.009375/0.00625/0.003125 axial/knight/corner weights.

EVIDENCE: Distance-stratified evaluation preserved 9,268 correct while reducing cross-entropy from 0.2124652 to 0.2124557 versus uniform outer-shift weights; the stronger redistribution lost one correct prediction, motivating alignment with the milder best-verified distribution.

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
            0.006250, 0.045000, 0.090000, 0.045000, 0.006250,
            0.009375, 0.090000, 0.360000, 0.090000, 0.009375,
            0.006250, 0.045000, 0.090000, 0.045000, 0.006250,
            0.003125, 0.006250, 0.009375, 0.006250, 0.003125,
        )
    )
>>>>>>> REPLACE