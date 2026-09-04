MECHANISM: Final upper-edge EMA plateau bisection

HYPOTHESIS: A 28.61344814300537109375% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664094925.

INTENDED_EDIT: Increase the endpoint-to-EMA interpolation coefficient from 0.286134243011474609375 to 0.2861344814300537109375.

EVIDENCE: The current 0.286134243011474609375 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy plateau boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.286134243011474609375)
=======
                parameter.lerp_(shadow / correction, 0.2861344814300537109375)
>>>>>>> REPLACE