MECHANISM: Bracketed vertical-shift TTA refinement

HYPOTHESIS: Vertical weights of 1.328125 will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.

INTENDED_EDIT: Restore the best-verified 0.98 EMA and test the midpoint between the best 1.3125/0.6875 TTA weights and the regressive 1.34375/0.65625 weights.

EVIDENCE: Correctness peaked at 9,348 with 1.3125 vertical weights and declined to 9,347 at both 1.34375 and 1.375, tightly bracketing the optimum above 1.3125.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (
            3.0,
            1.328125,
            1.328125,
            0.671875,
            0.671875,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                decay = 0.9825 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE