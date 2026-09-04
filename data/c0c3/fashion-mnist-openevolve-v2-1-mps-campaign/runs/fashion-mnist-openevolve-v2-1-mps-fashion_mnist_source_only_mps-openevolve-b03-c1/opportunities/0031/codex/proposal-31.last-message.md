MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.6163330078125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.286163330078125.

EVIDENCE: The current 0.2861328125 blend retains 9,133 correct, while 0.28619384765625 loses one correct but lowers cross-entropy; their midpoint is the most informative refinement of the accuracy plateau boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.286163330078125)
>>>>>>> REPLACE