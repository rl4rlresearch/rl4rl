MECHANISM: Slightly earlier BatchNorm-statistics alignment

HYPOTHESIS: Increasing the halfway-state buffer interpolation from 0.25 to 0.30 will exceed 9,260 correct predictions by better aligning BatchNorm statistics with the temporally averaged parameters.

INTENDED_EDIT: Preserve the best verified 0.03 parameter EMA and increase only the final floating-point buffer interpolation coefficient from 0.25 to 0.30.

EVIDENCE: The verified 0.25 mixture achieved 9,260 correct, whereas moving toward later buffers with 0.10 lost seven correct despite lower cross-entropy; because correct predictions dominate the objective, testing a conservative move in the opposite direction is the most informative next change.

<<<<<<< SEARCH
                    buffer.lerp_(start, 0.25)
=======
                    buffer.lerp_(start, 0.30)
>>>>>>> REPLACE