MECHANISM: Left-edge refinement of the flipped-view fusion plateau

HYPOTHESIS: A 0.508 flipped-view weight will preserve 9,331 correct predictions while lowering cross-entropy below the verified 0.509 weighting by reducing asymmetry.

INTENDED_EDIT: Change evaluation fusion from equal weighting to 0.492 original and 0.508 flipped while retaining the verified 1.184 calibration.

EVIDENCE: Flipped weight 0.509 achieved 9,331 correct with lower cross-entropy than 0.510 and 0.511, while 0.500 achieved 9,330; 0.508 is the closest untested point toward equal fusion and probes whether the accuracy gain persists with a better tie-breaker.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.492 * logits + 0.508 * flipped_logits)
>>>>>>> REPLACE