MECHANISM: Stronger flip-consistency-weighted logit ensembling

HYPOTHESIS: Increasing the flip-inconsistency penalty from 15% to 25% will exceed 9,289 correct predictions by further suppressing transformed offset pairs whose original and flipped predictions disagree.

INTENDED_EDIT: Change inconsistent offset-pair weights from 0.85 to 0.75 while leaving training, center preference, and agreement-conditioned calibration unchanged.

EVIDENCE: Moving from fixed pooling to a 15% flip-inconsistency penalty increased validation correctness from 9,287 to 9,289, while broader probability-space pooling fell to 9,286; a stronger dose of the successful targeted weighting is the most direct next test.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
=======
        offset_weights = base_weights * (0.75 + 0.25 * flip_consistency)
>>>>>>> REPLACE