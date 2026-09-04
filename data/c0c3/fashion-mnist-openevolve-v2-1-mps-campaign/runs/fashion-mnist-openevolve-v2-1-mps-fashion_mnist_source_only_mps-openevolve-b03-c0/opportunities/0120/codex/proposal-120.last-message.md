MECHANISM: Symmetric translated-logit optimum bracketing

HYPOTHESIS: A 37.48809814453125% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower verified setting, keeping weights complementary.

EVIDENCE: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.48779296875% was measurably worse; higher probes were equal or worse, so the untested lower midpoint most directly refines the optimum bracket.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.6251190185546875 * logits + 0.3748809814453125 * translated_logits
>>>>>>> REPLACE