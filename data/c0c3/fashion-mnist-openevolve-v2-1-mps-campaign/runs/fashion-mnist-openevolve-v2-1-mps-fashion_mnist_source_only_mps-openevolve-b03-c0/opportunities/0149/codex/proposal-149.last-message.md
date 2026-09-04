MECHANISM: Lower-side local-optimum retry

HYPOTHESIS: A 30.03330078125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Set the preserving blend to the previously unverified lower midpoint while retaining exactly complementary weights.

EVIDENCE: The current 0.300333203125 weight is best; 0.3003328125 was worse below it and 0.30033330078125 was worse above it. The intervening 0.3003330078125 probe timed out without validation evidence, making its retry the most informative unresolved test.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.6996669921875 * logits + 0.3003330078125 * translated_logits
>>>>>>> REPLACE