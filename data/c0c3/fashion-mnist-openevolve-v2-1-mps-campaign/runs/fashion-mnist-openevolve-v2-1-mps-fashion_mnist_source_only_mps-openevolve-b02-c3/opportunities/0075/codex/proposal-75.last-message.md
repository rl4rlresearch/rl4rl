MECHANISM: Fractional center-weight translation ensembling

HYPOTHESIS: A centered-crop weight of 2.5 with the verified 0.98 EMA will retain or exceed 9,344 correct predictions while improving cross-entropy relative to the weight-3 baseline.

INTENDED_EDIT: Restore the best-observed 0.98 EMA and interpolate the five-crop TTA center weight halfway between the weight-2 and weight-3 settings.

EVIDENCE: Weight 3 with decay 0.98 achieved 9,344 correct, while weight 2 achieved 9,343 but improved cross-entropy from 0.18781964 to 0.18779218; weight 4 sharply regressed to 9,338, motivating a focused midpoint toward the lower-cross-entropy side.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.5, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
                decay = 0.985 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE