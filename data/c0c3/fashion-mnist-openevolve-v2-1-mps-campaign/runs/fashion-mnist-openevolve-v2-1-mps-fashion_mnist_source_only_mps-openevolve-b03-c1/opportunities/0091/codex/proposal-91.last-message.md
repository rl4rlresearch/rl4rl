MECHANISM: Separable 3×3 translation-grid test-time ensemble

HYPOTHESIS: Adding the four missing diagonal ±1-pixel views with separable center/cardinal/diagonal weights will exceed 9,166 correct predictions by averaging coupled horizontal–vertical translation errors that the current axis-only ensemble cannot cover.

INTENDED_EDIT: Expand validation TTA from five to nine spatial views while retaining horizontal flips, logit-space aggregation, calibration scale, training behavior, and parameter count.

EVIDENCE: Probability-space aggregation reduced correctness from 9,166 to 9,163, supporting retention of weighted logit averaging; meanwhile, training independently augments both image axes but validation TTA covers only cardinal translations, motivating diagonal coverage instead of another aggregation change.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 2:30],
        )
        view_weights = (9.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0)
>>>>>>> REPLACE