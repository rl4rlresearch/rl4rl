MECHANISM: Train–test translation-kernel alignment

HYPOTHESIS: Matching training translations to the verified vertical-biased, distance-stratified TTA kernel will exceed 9,283 correct predictions by training the model on the same nuisance distribution used for inference averaging.

INTENDED_EDIT: Replace the symmetric training translation probabilities with the exact normalized TTA weights while preserving the best architecture, optimizer, loss, and schedule.

EVIDENCE: The best result used vertical-biased, distance-stratified TTA and reached 9,283 correct; nearby beta2, peak-rate, and cosine-floor changes regressed, while training augmentation remains mismatched at symmetric 0.09 axial weights and uniform radius-two weights.

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
            0.006250, 0.045000, 0.100000, 0.045000, 0.006250,
            0.009375, 0.080000, 0.360000, 0.080000, 0.009375,
            0.006250, 0.045000, 0.100000, 0.045000, 0.006250,
            0.003125, 0.006250, 0.009375, 0.006250, 0.003125,
        )
    )
>>>>>>> REPLACE