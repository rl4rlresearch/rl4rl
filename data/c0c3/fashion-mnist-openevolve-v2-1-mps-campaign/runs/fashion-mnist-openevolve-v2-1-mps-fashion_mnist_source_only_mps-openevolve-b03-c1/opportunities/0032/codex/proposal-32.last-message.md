MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.61480712890625% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2861480712890625.

EVIDENCE: The current 0.2861328125 blend retains 9,133 correct, while 0.286163330078125 loses one correct with lower cross-entropy; bisecting this interval is the most informative remaining test of the accuracy plateau’s upper boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.2861480712890625)
>>>>>>> REPLACE