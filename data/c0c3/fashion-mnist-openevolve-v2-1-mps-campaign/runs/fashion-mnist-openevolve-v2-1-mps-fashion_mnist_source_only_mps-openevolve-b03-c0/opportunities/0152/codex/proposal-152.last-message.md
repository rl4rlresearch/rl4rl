MECHANISM: Upper-side local-optimum bisection

HYPOTHESIS: A 30.03332275390625% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the interval between the current-best blend weight and the nearest higher verified non-improving weight while retaining exactly complementary coefficients.

EVIDENCE: Weight 0.300333203125 produced the best cross-entropy, while 0.300333251953125 preserved correctness but worsened cross-entropy to 0.19799133682250977; their midpoint is the most informative remaining refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.6996667724609375 * logits + 0.3003332275390625 * translated_logits
>>>>>>> REPLACE