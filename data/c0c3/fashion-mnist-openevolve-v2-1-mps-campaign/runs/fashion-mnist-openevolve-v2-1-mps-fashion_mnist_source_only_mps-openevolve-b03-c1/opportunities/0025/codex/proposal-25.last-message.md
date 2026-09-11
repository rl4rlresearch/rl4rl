MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.61328125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466473.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.28515625 to 0.2861328125.

EVIDENCE: The 28.515625% blend retained 9,133 correct, while 28.7109375% lost one correct but further reduced cross-entropy; bisecting this interval is the most informative test of the accuracy plateau’s upper boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28515625)
=======
                parameter.lerp_(shadow / correction, 0.2861328125)
>>>>>>> REPLACE