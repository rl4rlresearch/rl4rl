MECHANISM: Lower-bound translated-logit plateau refinement

HYPOTHESIS: A 37.48836517333984375% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower tested setting, preserving complementary weights and the 30% argmax-preserving blend.

EVIDENCE: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.4883270263671875% retained 9,322 correct but was measurably worse; their untested midpoint most directly refines the remaining lower boundary.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.6251163482666015625 * logits + 0.3748836517333984375 * translated_logits
>>>>>>> REPLACE