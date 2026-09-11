MECHANISM: Stronger flip-disagreement gating

HYPOTHESIS: Downweighting flip-inconsistent offsets by 12.5% while using the best verified calibration scale will exceed 9,325 correct predictions by reducing unreliable transformed evidence.

INTENDED_EDIT: Increase only the inference penalty for flip-inconsistent offsets from 10% to 12.5% and restore the verified 1.24754 calibration multiplier.

EVIDENCE: The uniform-pair hybrid ensemble achieved 9,325 correct at the best verified cross-entropy of 0.1929895668; the prior 12.5% gating attempt timed out, so it supplied no negative accuracy evidence and remains an unresolved compute-neutral inference refinement.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        offset_weights = base_weights * (0.875 + 0.125 * flip_consistency)
>>>>>>> REPLACE

<<<<<<< SEARCH
        calibration = 1.24003 * (
=======
        calibration = 1.24754 * (
>>>>>>> REPLACE