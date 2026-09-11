MECHANISM: Upper-plateau EMA boundary refinement

HYPOTHESIS: A 28.515625% EMA blend will retain 9,133 correct predictions while lowering cross-entropy below 0.2466727, producing a strict validation-score improvement.

INTENDED_EDIT: Set the endpoint-to-EMA interpolation coefficient to the midpoint between the best 28.125% blend and the accuracy-degrading 28.90625% blend.

EVIDENCE: The 28.125% blend achieved 9,133 correct, while 28.90625% achieved 9,132 with lower cross-entropy; bisecting this interval directly tests how much additional smoothing the best accuracy plateau permits.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28125)
=======
                parameter.lerp_(shadow / correction, 0.28515625)
>>>>>>> REPLACE