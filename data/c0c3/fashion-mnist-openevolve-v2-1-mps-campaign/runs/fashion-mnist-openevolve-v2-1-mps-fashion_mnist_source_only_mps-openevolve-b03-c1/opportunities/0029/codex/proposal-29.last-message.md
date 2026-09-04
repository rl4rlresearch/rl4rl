MECHANISM: Conservative upper-edge EMA plateau refinement

HYPOTHESIS: A 28.619384765625% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28619384765625.

EVIDENCE: The current 28.61328125% blend retains 9,133 correct, while 28.6376953125% loses one; the intervening 28.62548828125% test timed out without accuracy evidence, so bisecting its lower half cautiously advances toward the lower-cross-entropy boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.28619384765625)
>>>>>>> REPLACE