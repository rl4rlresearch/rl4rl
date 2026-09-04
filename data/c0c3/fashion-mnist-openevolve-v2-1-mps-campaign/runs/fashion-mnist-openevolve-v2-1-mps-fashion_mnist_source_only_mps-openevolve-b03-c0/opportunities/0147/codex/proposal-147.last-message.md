MECHANISM: Preserving-blend local optimum bisection

HYPOTHESIS: A 30.033330078125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the interval between the current-best blend weight and the higher verified non-improving weight, keeping the blend complementary.

EVIDENCE: The 0.300333203125 setting achieved the best cross-entropy of 0.1979913360595703, while 0.3003333984375 worsened it to 0.1979913375854492 with correctness unchanged; their midpoint is the most informative next probe.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.69966669921875 * logits + 0.30033330078125 * translated_logits
>>>>>>> REPLACE