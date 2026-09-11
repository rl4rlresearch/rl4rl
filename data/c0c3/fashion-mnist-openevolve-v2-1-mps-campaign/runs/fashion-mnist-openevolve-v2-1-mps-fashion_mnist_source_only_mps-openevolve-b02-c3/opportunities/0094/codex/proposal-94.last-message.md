MECHANISM: Boundary-aware vertical-shift TTA bisection

HYPOTHESIS: Vertical weights of 1.31640625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.

INTENDED_EDIT: Test the midpoint between the 9,348-correct 1.3125 weighting and the lower-cross-entropy but 9,347-correct 1.3203125 weighting, preserving center and total ensemble weight.

EVIDENCE: Weight 1.3125 achieved 9,348 correct, while 1.3203125 achieved 9,347 with slightly lower cross-entropy; their midpoint directly probes the decision boundary for a correctness-preserving cross-entropy gain.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.328125,
            1.328125,
            0.671875,
            0.671875,
        )
=======
        crop_weights = (
            3.0,
            1.31640625,
            1.31640625,
            0.68359375,
            0.68359375,
        )
>>>>>>> REPLACE