MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.313720703125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Test the midpoint between the successful 1.3134765625 weighting and the 9,347-correct 1.31396484375 weighting while preserving total ensemble weight.

EVIDENCE: Weight 1.3134765625 retained 9,348 correct with improved cross-entropy, whereas 1.31396484375 lost one prediction despite another small cross-entropy reduction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.313720703125,
            1.313720703125,
            0.686279296875,
            0.686279296875,
        )
>>>>>>> REPLACE