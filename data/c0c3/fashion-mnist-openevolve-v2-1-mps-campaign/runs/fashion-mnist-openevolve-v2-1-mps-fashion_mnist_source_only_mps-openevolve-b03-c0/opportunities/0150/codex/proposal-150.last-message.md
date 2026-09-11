MECHANISM: Upper-side local-optimum bisection

HYPOTHESIS: A 30.0333251953125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the interval between the current-best blend weight and the nearest higher verified non-improving weight, retaining exactly complementary weights.

EVIDENCE: Weight 0.300333203125 achieved the best cross-entropy of 0.1979913360595703, while 0.30033330078125 worsened it to 0.19799133911132813 with correctness unchanged; their untested midpoint is the most informative remaining local refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.699666748046875 * logits + 0.300333251953125 * translated_logits
>>>>>>> REPLACE