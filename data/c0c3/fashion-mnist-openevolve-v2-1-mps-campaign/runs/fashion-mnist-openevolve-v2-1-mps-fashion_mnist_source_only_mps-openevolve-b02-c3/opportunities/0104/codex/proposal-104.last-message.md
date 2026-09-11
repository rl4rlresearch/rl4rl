MECHANISM: Conservative correctness-boundary TTA refinement

HYPOTHESIS: Vertical weights of 1.3134918212890625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

INTENDED_EDIT: Restore the best-performing TTA neighborhood and test a conservative quarter-step above the verified 1.3134765625 weight, preserving center and total ensemble weight.

EVIDENCE: Weight 1.3134765625 achieved 9,348 correct, while 1.31353759765625 lost one prediction; the intervening midpoint timed out without producing evidence, so this lower-side probe offers the best chance of a correctness-preserving cross-entropy gain.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.314453125,
            1.314453125,
            0.685546875,
            0.685546875,
        )
=======
        crop_weights = (
            3.0,
            1.3134918212890625,
            1.3134918212890625,
            0.6865081787109375,
            0.6865081787109375,
        )
>>>>>>> REPLACE