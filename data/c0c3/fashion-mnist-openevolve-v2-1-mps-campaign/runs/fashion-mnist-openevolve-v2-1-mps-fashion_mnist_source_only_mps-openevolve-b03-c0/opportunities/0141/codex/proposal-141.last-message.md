MECHANISM: Preserving-blend boundary bisection

HYPOTHESIS: A 30.0334375% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799134407043456.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.30033125 to 0.300334375 while keeping the weights complementary.

EVIDENCE: The verified 30.033125% setting is the current best, while the higher 30.03375% probe timed out without contrary validation evidence; bisecting that interval is the most conservative informative refinement.

<<<<<<< SEARCH
            preserving_logits = 0.69966875 * logits + 0.30033125 * translated_logits
=======
            preserving_logits = 0.699665625 * logits + 0.300334375 * translated_logits
>>>>>>> REPLACE