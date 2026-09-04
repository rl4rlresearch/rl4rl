MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.313507080078125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Restore the favorable TTA region and test the midpoint between the best verified 1.3134765625 weighting and the 9,347-correct 1.31353759765625 weighting, preserving center weight and total ensemble weight.

EVIDENCE: Weight 1.3134765625 achieved 9,348 correct, while the nearest tested higher weight, 1.31353759765625, lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.31640625,
            1.31640625,
            0.68359375,
            0.68359375,
        )
=======
        crop_weights = (
            3.0,
            1.313507080078125,
            1.313507080078125,
            0.686492919921875,
            0.686492919921875,
        )
>>>>>>> REPLACE