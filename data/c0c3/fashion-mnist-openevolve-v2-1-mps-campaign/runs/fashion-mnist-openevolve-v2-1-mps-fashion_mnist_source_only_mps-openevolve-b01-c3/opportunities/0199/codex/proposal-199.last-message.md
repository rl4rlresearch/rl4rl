MECHANISM: Continued flip-disagreement reliability gating

HYPOTHESIS: Increasing the flip-inconsistency penalty from 20% to 22.5% will preserve at least 9,326 correct predictions and reduce validation cross-entropy below 0.1929811478.

INTENDED_EDIT: Reduce only the inference weight assigned to flip-inconsistent offsets from 80% to 77.5%, leaving training, hybrid pooling, and calibration unchanged.

EVIDENCE: Penalties from 10% through 20% successively reduced cross-entropy, while 17.5% raised correctness to 9,326 and 20% retained it; a further 2.5-point continuation is the most direct local test.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        offset_weights = base_weights * (0.775 + 0.225 * flip_consistency)
>>>>>>> REPLACE