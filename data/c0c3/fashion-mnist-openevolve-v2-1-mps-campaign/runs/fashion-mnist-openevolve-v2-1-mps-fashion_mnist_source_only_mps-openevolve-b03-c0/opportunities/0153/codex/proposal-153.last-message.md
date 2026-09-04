MECHANISM: Lower-side local-optimum bisection

HYPOTHESIS: A 30.033310546875% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the unresolved lower interval around the current-best preserving blend while retaining exactly complementary coefficients.

EVIDENCE: Weight 0.300333203125 produced the best cross-entropy, while the lower 0.3003330078125 setting worsened it to 0.19799134063720703; their untested midpoint is the most informative remaining lower-side refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.69966689453125 * logits + 0.30033310546875 * translated_logits
>>>>>>> REPLACE