MECHANISM: Reduced label-smoothing regularization with verified logit calibration

HYPOTHESIS: Halving label smoothing to 0.01 will improve validation_correct above 9,256 by strengthening class margins during the limited training exposure, while the verified 1.35605 multiplier will minimize cross-entropy without changing argmax predictions.

INTENDED_EDIT: Reduce training label smoothing from 0.02 to 0.01 and replace the evaluation multiplier with the best verified temperature.

EVIDENCE: Changes to augmentation duration and evaluation-view weighting consistently reduced validation_correct, motivating an orthogonal loss-side refinement; Reference Design 2 verified 1.35605 as the lowest-cross-entropy multiplier at the 9,256-correct baseline.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        label_smoothing=0.02,
=======
        label_smoothing=0.01,
>>>>>>> REPLACE