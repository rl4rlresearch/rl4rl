MECHANISM: Boundary-bisecting flip-consistency weighting

HYPOTHESIS: A 9.6875% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below the 10% penalty’s 0.20669593772888184.

INTENDED_EDIT: Increase the current penalty from 7.5% to 9.6875%, bisecting the established boundary between the unsuccessful 9.375% dose and successful 10% dose.

EVIDENCE: The 10% penalty achieved 9,289 correct, while 9.375% achieved 9,288; 9.6875% is the most informative untested point between them.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.925 + 0.075 * flip_consistency)
=======
        offset_weights = base_weights * (
            0.903125 + 0.096875 * flip_consistency
        )
>>>>>>> REPLACE