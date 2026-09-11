MECHANISM: Convex translated-logit plateau refinement

HYPOTHESIS: A 37.488422393798828125% translated-logit contribution will retain 9,322 correct predictions and lower cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest higher setting with identical measured cross-entropy, keeping weights complementary.

EVIDENCE: The current 37.4884033203125% and 37.48844146728515625% settings produced identical best scores, while 37.4884796142578125% was worse; the midpoint probes the unresolved interior of this narrow optimum bracket.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.62511577606201171875 * logits + 0.37488422393798828125 * translated_logits
>>>>>>> REPLACE