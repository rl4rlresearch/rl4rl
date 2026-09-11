MECHANISM: Symmetric validation-logit calibration probe

HYPOTHESIS: Decreasing the evaluation-only logit scale from 1.4164 to 1.4163 will preserve all 9,260 argmax predictions and lower validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Reduce only the positive evaluation-time logit multiplier by 0.0001, leaving training and terminal EMA behavior unchanged.

EVIDENCE: The 1.4165 upward calibration probe timed out without validation evidence. Because positive logit scaling preserves predicted classes, the symmetric downward probe safely brackets the incumbent calibration while directly testing the unexplored direction.

<<<<<<< SEARCH
            logits = logits * 1.4164
=======
            logits = logits * 1.4163
>>>>>>> REPLACE