MECHANISM: Half-step canonical-view-biased flip logit ensemble

HYPOTHESIS: A 52.5% original / 47.5% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.18488420372009276.

INTENDED_EDIT: Bias evaluation logits halfway toward the tested 55% canonical-view weighting while leaving training and calibration unchanged.

EVIDENCE: The 55% original weighting lowered cross-entropy substantially to 0.18461807746887207 but lost one correct prediction; testing the midpoint determines whether part of that calibration gain occurs before the accuracy boundary.

<<<<<<< SEARCH
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        return 1.226016 * (
            0.525 * logits + 0.475 * flipped_logits
        )
>>>>>>> REPLACE