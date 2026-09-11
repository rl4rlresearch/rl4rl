MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.662109375% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664096.

INTENDED_EDIT: Increase the endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28662109375.

EVIDENCE: The 28.61328125% blend retained 9,133 correct, while 28.7109375% lost one correct but achieved lower cross-entropy; their midpoint is the most informative refinement of the accuracy plateau’s upper boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.28662109375)
>>>>>>> REPLACE