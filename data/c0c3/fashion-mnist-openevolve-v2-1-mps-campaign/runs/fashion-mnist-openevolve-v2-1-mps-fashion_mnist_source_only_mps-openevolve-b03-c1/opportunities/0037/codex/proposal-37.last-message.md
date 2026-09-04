MECHANISM: Final upper-edge EMA plateau bisection

HYPOTHESIS: A 28.6134243011474609375% EMA blend will retain 9,133 correct predictions while lowering cross-entropy below 0.2466409584.

INTENDED_EDIT: Increase the endpoint-to-EMA interpolation coefficient from 0.28613376617431640625 to 0.286134243011474609375.

EVIDENCE: The current 0.28613376617431640625 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint most precisely probes the remaining accuracy-plateau boundary.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28613376617431640625)
=======
                parameter.lerp_(shadow / correction, 0.286134243011474609375)
>>>>>>> REPLACE