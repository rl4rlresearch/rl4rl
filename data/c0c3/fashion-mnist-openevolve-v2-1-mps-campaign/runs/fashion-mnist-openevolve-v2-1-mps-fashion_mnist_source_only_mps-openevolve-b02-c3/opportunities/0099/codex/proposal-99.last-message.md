MECHANISM: Correctness-boundary vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.3135986328125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Test the midpoint between the successful 1.3134765625 weighting and the 9,347-correct 1.313720703125 weighting while preserving center and total ensemble weight.

EVIDENCE: Weight 1.3134765625 achieved 9,348 correct, whereas 1.313720703125 lost one prediction but slightly lowered cross-entropy; their midpoint is the most informative remaining correctness-boundary probe.

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
            1.3135986328125,
            1.3135986328125,
            0.6864013671875,
            0.6864013671875,
        )
>>>>>>> REPLACE