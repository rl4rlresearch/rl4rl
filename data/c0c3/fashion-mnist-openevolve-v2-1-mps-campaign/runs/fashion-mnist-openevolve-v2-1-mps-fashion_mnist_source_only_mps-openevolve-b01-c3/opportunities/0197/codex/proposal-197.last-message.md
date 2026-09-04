MECHANISM: Continued flip-disagreement reliability gating

HYPOTHESIS: Increasing the flip-inconsistency penalty from 15% to 17.5% will preserve 9,325 correct predictions while reducing validation cross-entropy below 0.1929848656.

INTENDED_EDIT: Keep training, hybrid pooling, and calibration unchanged; reduce only the inference weight of flip-inconsistent offsets from 85% to 82.5%.

EVIDENCE: Increasing the penalty from 10% to 12.5% and then 15% successively reduced cross-entropy from 0.1929895668 to 0.1929871063 and 0.1929848656 without changing the 9,325 correct predictions.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
=======
        offset_weights = base_weights * (0.825 + 0.175 * flip_consistency)
>>>>>>> REPLACE