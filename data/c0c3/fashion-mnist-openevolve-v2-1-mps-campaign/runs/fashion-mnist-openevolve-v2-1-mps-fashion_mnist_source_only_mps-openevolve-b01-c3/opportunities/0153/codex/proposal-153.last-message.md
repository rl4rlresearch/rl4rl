MECHANISM: Boundary-bisecting flip-consistency weighting

HYPOTHESIS: A 9.84375% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.

INTENDED_EDIT: Reduce the inconsistent offset-pair penalty from 10% to 9.84375%, leaving training, center preference, and agreement calibration unchanged.

EVIDENCE: A 10% penalty retained 9,289 correct, whereas 9.6875% produced 9,288; 9.84375% bisects this established decision boundary and is the closest untested lower penalty likely to preserve correctness while improving calibration.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        offset_weights = base_weights * (
            0.9015625 + 0.0984375 * flip_consistency
        )
>>>>>>> REPLACE