MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.614044189453125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28614044189453125.

EVIDENCE: The current 0.2861328125 blend retains 9,133 correct, while 0.2861480712890625 loses one correct with slightly lower cross-entropy; bisecting this interval most directly refines the accuracy plateau’s upper boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.28614044189453125)
>>>>>>> REPLACE