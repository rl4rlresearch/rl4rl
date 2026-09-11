MECHANISM: Directional vertical-shift TTA reweighting

HYPOTHESIS: The verified asymmetric crop weights will retain 9,348 correct predictions while lowering cross-entropy below the current 0.187707582855.

INTENDED_EDIT: Restore the best-verified ensemble weights by transferring 1/1024 weight between vertical crops and preserving total weight.

EVIDENCE: Reference Design 1 achieved 9,348 correct with cross-entropy 0.187706897736, the highest available validation_score.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.3125,
            1.314453125,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE