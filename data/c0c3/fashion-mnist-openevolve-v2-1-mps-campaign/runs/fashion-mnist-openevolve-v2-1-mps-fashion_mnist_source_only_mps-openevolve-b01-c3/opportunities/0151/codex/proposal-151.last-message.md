MECHANISM: Boundary-bisecting flip-consistency weighting

HYPOTHESIS: A 9.375% flip-inconsistency penalty will retain 9,289 correct predictions while reducing validation cross-entropy below the 10% penalty’s 0.20669593772888184.

INTENDED_EDIT: Reduce the inconsistent offset-pair penalty from 15% to 9.375%, leaving training, center preference, and agreement calibration unchanged.

EVIDENCE: Penalties of 10%–15% retained 9,289 correct, while 8.75% lost one; 9.375% bisects the established decision boundary and tests whether the lower-cross-entropy dose can preserve the accuracy gain.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
=======
        offset_weights = base_weights * (0.90625 + 0.09375 * flip_consistency)
>>>>>>> REPLACE