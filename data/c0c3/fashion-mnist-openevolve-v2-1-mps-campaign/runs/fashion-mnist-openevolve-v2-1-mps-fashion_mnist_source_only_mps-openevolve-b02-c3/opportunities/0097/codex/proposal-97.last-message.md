MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.31396484375 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Test the midpoint between the 9,348-correct 1.3134765625 weighting and the 9,347-correct 1.314453125 weighting, preserving center weight and total ensemble weight.

EVIDENCE: Increasing the vertical weight from 1.3125 to 1.3134765625 retained 9,348 correct and reduced cross-entropy, while 1.314453125 lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3134765625,
            1.3134765625,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.31396484375,
            1.31396484375,
            0.68603515625,
            0.68603515625,
        )
>>>>>>> REPLACE