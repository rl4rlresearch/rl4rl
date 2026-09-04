MECHANISM: Flipped-view-biased logit fusion

HYPOTHESIS: Weighting the flipped validation view 0.55 and the original view 0.45 will exceed 9,330 correct predictions, because the opposite 0.55 original-view bias reduced correctness by ten.

INTENDED_EDIT: Replace equal evaluation-time fusion with a slight flipped-view bias while retaining the verified 1.184 calibration.

EVIDENCE: Equal fusion verified 9,330 correct, while the 0.55 original/0.45 flipped trial fell to 9,320; this establishes a directional signal favoring the previously downweighted flipped view and motivates the mirrored test.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.45 * logits + 0.55 * flipped_logits)
>>>>>>> REPLACE