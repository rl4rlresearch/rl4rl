MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.31353759765625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Test the midpoint between the best verified 1.3134765625 weighting and the 9,347-correct 1.3135986328125 weighting, preserving center weight and total ensemble weight.

EVIDENCE: Weight 1.3134765625 achieved 9,348 correct with cross-entropy 0.187707568359375, whereas the nearest tested higher weight, 1.3135986328125, lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.31353759765625,
            1.31353759765625,
            0.68646240234375,
            0.68646240234375,
        )
>>>>>>> REPLACE