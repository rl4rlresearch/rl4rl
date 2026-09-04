MECHANISM: Boundary-bisecting flip-consistency weighting

HYPOTHESIS: A 9.921875% flip-inconsistency penalty will retain 9,289 correct predictions while lowering cross-entropy below the 10% penalty’s 0.20669593772888184.

INTENDED_EDIT: Reduce the current 12.5% penalty to 9.921875%, leaving training, center preference, and agreement calibration unchanged.

EVIDENCE: A 10% penalty achieved 9,289 correct, while 9.84375% achieved 9,288; 9.921875% bisects the established correctness boundary and is the most informative lower penalty likely to retain the extra correct prediction.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.875 + 0.125 * flip_consistency)
=======
        offset_weights = base_weights * (
            0.90078125 + 0.09921875 * flip_consistency
        )
>>>>>>> REPLACE