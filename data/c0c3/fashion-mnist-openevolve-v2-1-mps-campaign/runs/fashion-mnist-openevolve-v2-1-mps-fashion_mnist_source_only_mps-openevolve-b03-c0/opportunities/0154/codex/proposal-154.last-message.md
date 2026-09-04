MECHANISM: Lower-side local-optimum bisection

HYPOTHESIS: A 30.0333154296875% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the interval between the current-best blend weight and the nearest lower verified non-improving weight, retaining exactly complementary coefficients.

EVIDENCE: Weight 0.300333203125 achieved the best cross-entropy, while the lower 0.30033310546875 setting preserved correctness but worsened cross-entropy to 0.1979913375854492; their midpoint is the most informative remaining lower-side refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.699666845703125 * logits + 0.300333154296875 * translated_logits
>>>>>>> REPLACE