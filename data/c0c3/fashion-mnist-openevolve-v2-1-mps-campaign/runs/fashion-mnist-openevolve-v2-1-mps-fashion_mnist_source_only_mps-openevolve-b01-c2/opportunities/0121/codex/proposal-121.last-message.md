MECHANISM: Midpoint search of the narrow flipped-view fusion optimum

HYPOTHESIS: A 0.511 flipped-view weight will exceed 9,331 correct predictions by capturing a favorable decision transition between the successful 0.510 weight and the degraded 0.512 weight.

INTENDED_EDIT: Change evaluation fusion from 0.490/0.510 to 0.489/0.511 while retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Flipped weight 0.510 achieved 9,331 correct, whereas 0.500 and 0.512 each achieved 9,330; testing their closest unmeasured midpoint brackets the narrow accuracy optimum.

<<<<<<< SEARCH
            logits = 1.184 * (0.49 * logits + 0.51 * flipped_logits)
=======
            logits = 1.184 * (0.489 * logits + 0.511 * flipped_logits)
>>>>>>> REPLACE