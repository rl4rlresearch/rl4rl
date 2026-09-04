MECHANISM: Lower-side blend-weight bisection

HYPOTHESIS: A 30.03331787109375% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Bisect the remaining interval between the current-best preserving blend and the nearest lower verified non-improving blend, keeping coefficients complementary.

EVIDENCE: Weight 0.300333203125 achieved the best cross-entropy, while 0.300333154296875 preserved correctness but worsened cross-entropy to 0.19799133949279785; their untested midpoint is the most informative remaining local refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.6996668212890625 * logits + 0.3003331787109375 * translated_logits
>>>>>>> REPLACE