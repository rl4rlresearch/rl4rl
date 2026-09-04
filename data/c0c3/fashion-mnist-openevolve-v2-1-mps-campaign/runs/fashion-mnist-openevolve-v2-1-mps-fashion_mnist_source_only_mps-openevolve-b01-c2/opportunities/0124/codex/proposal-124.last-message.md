MECHANISM: Left-edge refinement of the flipped-view fusion plateau

HYPOTHESIS: A 0.5085 flipped-view weight will preserve 9,331 correct predictions while reducing cross-entropy below the verified 0.509 weighting through slightly less asymmetric fusion.

INTENDED_EDIT: Change evaluation-time fusion from equal weighting to 0.4915 original and 0.5085 flipped, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: A 0.509 flipped weight achieved 9,331 correct with 0.196251798 cross-entropy, outperforming 0.510 and 0.511 on the tie-breaker; 0.500 lost one prediction, while the direct 0.508 trial timed out without performance evidence, motivating a conservative midpoint toward the unresolved plateau edge.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.4915 * logits + 0.5085 * flipped_logits)
>>>>>>> REPLACE