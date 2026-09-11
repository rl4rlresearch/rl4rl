MECHANISM: Boundary-aware vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.314453125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.

INTENDED_EDIT: Restore the favorable TTA region and test the midpoint between the 9,348-correct 1.3125 weighting and the 9,347-correct 1.31640625 weighting, preserving total ensemble weight.

EVIDENCE: Weight 1.3125 achieved 9,348 correct, while 1.31640625 lost one prediction but reduced cross-entropy to 0.187707538986; their midpoint most directly locates the correctness boundary and tests for a tie-breaking cross-entropy gain.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.34375, 1.34375, 0.65625, 0.65625)
=======
        crop_weights = (
            3.0,
            1.314453125,
            1.314453125,
            0.685546875,
            0.685546875,
        )
>>>>>>> REPLACE