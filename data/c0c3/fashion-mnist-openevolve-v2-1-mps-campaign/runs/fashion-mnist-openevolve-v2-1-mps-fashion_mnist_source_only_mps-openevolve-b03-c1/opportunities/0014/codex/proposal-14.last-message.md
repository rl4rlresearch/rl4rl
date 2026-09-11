MECHANISM: Midpoint endpoint–EMA weight interpolation

HYPOTHESIS: Increasing the EMA interpolation from 25% to 50% will exceed 9,130 correct predictions by capturing more trajectory smoothing without the accuracy loss caused by full EMA replacement.

INTENDED_EDIT: Blend final parameters halfway toward the bias-corrected 0.99 EMA instead of one-quarter toward it.

EVIDENCE: The endpoint achieved 9,122 correct, 25% EMA blending improved this to 9,130, and full EMA achieved 9,121 with substantially lower cross-entropy; these results indicate an interior blend is best and motivate testing the midpoint.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.25)
=======
                parameter.lerp_(shadow / correction, 0.50)
>>>>>>> REPLACE