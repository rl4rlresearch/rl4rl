MECHANISM: Upper-side local-optimum retry

HYPOTHESIS: A 30.0333251953125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Set the preserving blend to the unverified midpoint between the current-best weight and the nearest higher verified non-improving weight, retaining exactly complementary coefficients.

EVIDENCE: Weight 0.300333203125 achieved the best cross-entropy, while 0.30033330078125 worsened it with correctness unchanged; the intervening midpoint previously timed out and remains the most informative unresolved probe.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.699666748046875 * logits + 0.300333251953125 * translated_logits
>>>>>>> REPLACE