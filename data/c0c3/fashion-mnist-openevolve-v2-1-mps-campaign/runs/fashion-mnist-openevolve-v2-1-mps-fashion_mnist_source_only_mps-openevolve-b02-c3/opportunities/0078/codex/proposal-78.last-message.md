MECHANISM: Near-center translation-ensemble boundary refinement

HYPOTHESIS: A centered-crop weight of 2.875 with the verified 0.98 EMA will retain 9,344 correct predictions while lowering cross-entropy below 0.18781964.

INTENDED_EDIT: Restore the best-observed EMA decay and test the midpoint between centered-crop weights 2.75 and 3.0.

EVIDENCE: With decay 0.98, weight 3.0 achieved 9,344 correct, while weight 2.75 lowered cross-entropy but lost two predictions; 2.875 directly refines that unresolved accuracy-calibration boundary.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.875, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
                decay = 0.9825 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE