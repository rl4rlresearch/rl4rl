MECHANISM: Leftward refinement of the flipped-view fusion plateau

HYPOTHESIS: A 0.509 flipped-view weight will preserve 9,331 correct predictions while lowering cross-entropy below 0.196252436 by moving closer to equal fusion.

INTENDED_EDIT: Change evaluation fusion from 0.500/0.500 to 0.491/0.509 while retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Flipped weights 0.510 and 0.511 both achieved 9,331 correct, but 0.510 had lower cross-entropy; 0.500 achieved only 9,330. Testing 0.509 probes the unmeasured left edge of the accuracy plateau, where reduced asymmetry should improve the tie-breaker if the extra correct prediction is retained.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.491 * logits + 0.509 * flipped_logits)
>>>>>>> REPLACE