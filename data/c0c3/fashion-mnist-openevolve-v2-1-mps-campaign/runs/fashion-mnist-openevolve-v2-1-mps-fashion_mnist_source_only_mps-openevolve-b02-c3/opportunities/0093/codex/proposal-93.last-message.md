MECHANISM: Quadratic-bracketed vertical-shift TTA refinement

HYPOTHESIS: Vertical weights of 1.3203125 will retain at least 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.

INTENDED_EDIT: Test the midpoint between the best 1.3125/0.6875 weights and the narrowly regressive 1.328125/0.671875 weights, preserving center and total ensemble weight.

EVIDENCE: The 1.3125 design achieved 9,348 correct; 1.328125 lost one prediction but had nearly identical cross-entropy, while 1.34375 worsened cross-entropy more clearly. Their cross-entropy trend places the interpolated minimum near this midpoint.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (
            3.0,
            1.3203125,
            1.3203125,
            0.6796875,
            0.6796875,
        )
>>>>>>> REPLACE