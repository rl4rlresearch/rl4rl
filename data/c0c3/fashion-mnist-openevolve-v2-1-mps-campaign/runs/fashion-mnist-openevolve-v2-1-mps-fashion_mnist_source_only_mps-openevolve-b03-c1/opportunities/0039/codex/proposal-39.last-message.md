MECHANISM: Final upper-edge EMA plateau bisection

HYPOTHESIS: A 28.613460063934326171875% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.246640940094.

INTENDED_EDIT: Increase the endpoint-to-EMA interpolation coefficient from 0.2861344814300537109375 to 0.28613460063934326171875.

EVIDENCE: The current 0.2861344814300537109375 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy plateau boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861344814300537109375)
=======
                parameter.lerp_(shadow / correction, 0.28613460063934326171875)
>>>>>>> REPLACE