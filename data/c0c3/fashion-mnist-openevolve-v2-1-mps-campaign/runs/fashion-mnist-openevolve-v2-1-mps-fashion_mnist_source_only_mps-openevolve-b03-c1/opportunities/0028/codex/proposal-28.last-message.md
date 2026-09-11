MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.62548828125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664096.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2862548828125.

EVIDENCE: The 28.61328125% blend retained 9,133 correct, while 28.6376953125% lost one correct but lowered cross-entropy; their midpoint directly refines the accuracy plateau’s upper boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.2862548828125)
>>>>>>> REPLACE