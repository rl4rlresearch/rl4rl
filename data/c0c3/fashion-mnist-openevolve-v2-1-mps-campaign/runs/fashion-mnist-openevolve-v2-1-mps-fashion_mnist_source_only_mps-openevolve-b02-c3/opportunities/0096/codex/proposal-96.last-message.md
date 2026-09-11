MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.3134765625 will retain 9,348 correct predictions while lowering cross-entropy below 0.187707582855.

INTENDED_EDIT: Test the midpoint between the 9,348-correct 1.3125 weighting and the lower-cross-entropy but 9,347-correct 1.314453125 weighting, preserving total ensemble weight.

EVIDENCE: Weight 1.3125 achieved 9,348 correct, while 1.314453125 lost one prediction but lowered cross-entropy to 0.187707558441; their midpoint is the most informative remaining probe of the correctness boundary.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.25, 1.25, 0.75, 0.75)
=======
        crop_weights = (
            3.0,
            1.3134765625,
            1.3134765625,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE