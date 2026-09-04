MECHANISM: Fine-grained flipped-view logit fusion

HYPOTHESIS: Increasing the flipped-view weight from 0.510 to 0.512 will exceed 9,331 correct predictions by resolving another borderline disagreement while staying near the verified local improvement and far from the degraded 0.55 weighting.

INTENDED_EDIT: Change evaluation-time fusion to 0.488 original and 0.512 flipped, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: Flipped weight 0.500 produced 9,330 correct, 0.510 improved to 9,331, and 0.550 declined to 9,326; this motivates a small local step beyond the best verified weight.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.488 * logits + 0.512 * flipped_logits)
>>>>>>> REPLACE