MECHANISM: Lower-side translated-logit optimum refinement

HYPOTHESIS: A 37.4883270263671875% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower setting with worse measured cross-entropy; keep complementary weights and the 30% argmax-preserving blend unchanged.

EVIDENCE: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.488250732421875% retained the same correct count but was worse; their untested midpoint most directly refines the remaining lower-side interval.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.625116729736328125 * logits + 0.374883270263671875 * translated_logits
>>>>>>> REPLACE