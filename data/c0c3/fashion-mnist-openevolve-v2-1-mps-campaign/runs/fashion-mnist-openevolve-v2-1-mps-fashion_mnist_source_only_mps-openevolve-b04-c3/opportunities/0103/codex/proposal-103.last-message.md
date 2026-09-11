MECHANISM: Quadratic inference-temperature refinement

HYPOTHESIS: Scaling ensemble logits by 1.035946 will preserve all 9,243 predictions and reduce validation cross-entropy below 0.20883300514221192.

INTENDED_EDIT: Change only the positive inference-time logit scale from 1.03577 to 1.035946.

EVIDENCE: The closest verified scales—1.035897, 1.03592, and 1.0360—give a quadratic minimum near 1.035946; its prior verification timed out and therefore provides no contradictory performance evidence.

<<<<<<< SEARCH
            logits = 1.03577 * (
=======
            logits = 1.035946 * (
>>>>>>> REPLACE