MECHANISM: Boundary-seeking flip-consistency weighting

HYPOTHESIS: An 8.75% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.

INTENDED_EDIT: Reduce the inconsistent offset-pair penalty from 10% to 8.75%, leaving training, center preference, and agreement calibration unchanged.

EVIDENCE: Penalties of 15%, 12.5%, and 10% all retained 9,289 correct while lower penalties progressively improved cross-entropy; 7.5% lost one correct prediction, placing the useful decision boundary between 7.5% and 10%.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        offset_weights = base_weights * (0.9125 + 0.0875 * flip_consistency)
>>>>>>> REPLACE