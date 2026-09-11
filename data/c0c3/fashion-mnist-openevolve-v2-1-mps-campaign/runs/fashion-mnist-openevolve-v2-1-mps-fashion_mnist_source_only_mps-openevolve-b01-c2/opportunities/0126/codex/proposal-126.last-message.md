MECHANISM: Conservative left-edge refinement of flipped-view fusion

HYPOTHESIS: A 0.50875 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251798 by moving toward equal fusion.

INTENDED_EDIT: Change evaluation fusion from 0.490/0.510 to 0.49125/0.50875 while retaining the verified 1.184 calibration.

EVIDENCE: Weights 0.509 and 0.510 both achieved 9,331 correct, with 0.509 producing lower cross-entropy; the unresolved 0.5085 trial motivates testing the conservative midpoint between it and the verified 0.509 design.

<<<<<<< SEARCH
            logits = 1.184 * (0.49 * logits + 0.51 * flipped_logits)
=======
            logits = 1.184 * (0.49125 * logits + 0.50875 * flipped_logits)
>>>>>>> REPLACE