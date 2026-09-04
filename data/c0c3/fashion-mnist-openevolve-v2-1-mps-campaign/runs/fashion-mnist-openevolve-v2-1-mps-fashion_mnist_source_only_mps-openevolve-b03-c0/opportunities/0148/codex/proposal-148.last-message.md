MECHANISM: Lower-side local-optimum bisection

HYPOTHESIS: A 30.03330078125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the interval between the current-best blend weight and the lower verified weight, keeping the blend weights complementary.

EVIDENCE: Cross-entropy improved from 0.19799134063720703 at 0.3003328125 to 0.1979913360595703 at 0.300333203125, then worsened to 0.19799133911132813 at 0.30033330078125; probing the untested lower midpoint most directly refines the bracketed local optimum.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.6996669921875 * logits + 0.3003330078125 * translated_logits
>>>>>>> REPLACE