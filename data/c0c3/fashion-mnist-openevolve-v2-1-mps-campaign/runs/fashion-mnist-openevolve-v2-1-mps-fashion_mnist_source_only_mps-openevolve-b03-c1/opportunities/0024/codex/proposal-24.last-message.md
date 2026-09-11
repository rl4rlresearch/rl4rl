MECHANISM: Upper-edge EMA plateau bisection

HYPOTHESIS: A 28.7109375% EMA blend will retain 9,133 correct predictions while reducing cross-entropy below 0.2466473.

INTENDED_EDIT: Increase the final endpoint-to-EMA interpolation coefficient from 0.28515625 to 0.287109375.

EVIDENCE: The 28.515625% blend retained 9,133 correct with lower cross-entropy than 28.125%, while 28.90625% lost one correct but lowered cross-entropy further; their midpoint tests the accuracy plateau’s upper edge.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28515625)
=======
                parameter.lerp_(shadow / correction, 0.287109375)
>>>>>>> REPLACE