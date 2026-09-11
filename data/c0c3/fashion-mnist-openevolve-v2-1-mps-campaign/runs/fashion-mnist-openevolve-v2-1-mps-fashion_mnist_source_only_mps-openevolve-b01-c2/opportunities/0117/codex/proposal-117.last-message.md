MECHANISM: Interpolated flipped-view logit fusion

HYPOTHESIS: A 0.516 flipped-view weight will exceed 9,331 correct predictions by moving slightly beyond the successful 0.51 weight while remaining far from the degraded 0.55 setting.

INTENDED_EDIT: Change evaluation fusion from 0.49/0.51 to 0.484/0.516, retaining the verified 1.184 calibration.

EVIDENCE: Equal fusion produced 9,330 correct, 0.51 flipped weighting improved to 9,331, and 0.55 fell to 9,326; these results place the most promising untested region just beyond 0.51.

<<<<<<< SEARCH
            logits = 1.184 * (0.49 * logits + 0.51 * flipped_logits)
=======
            logits = 1.184 * (0.484 * logits + 0.516 * flipped_logits)
>>>>>>> REPLACE