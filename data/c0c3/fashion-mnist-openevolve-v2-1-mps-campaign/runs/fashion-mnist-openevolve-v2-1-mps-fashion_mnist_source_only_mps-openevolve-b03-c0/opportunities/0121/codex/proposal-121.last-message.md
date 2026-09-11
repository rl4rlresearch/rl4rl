MECHANISM: Interior translated-logit optimum probe

HYPOTHESIS: A 37.488250732421875% translated-logit contribution will retain 9,322 correct predictions while lowering cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Set the unanimous-correction blend to the midpoint of the two distinct settings that achieved the best measured cross-entropy, keeping complementary weights and the 30% argmax-preserving blend unchanged.

EVIDENCE: Both 37.4884033203125% and 37.48809814453125% produced the identical best score, while the nearest tested settings outside that interval were worse; its untested midpoint is the most informative convex-optimum probe.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.62511749267578125 * logits + 0.37488250732421875 * translated_logits
>>>>>>> REPLACE