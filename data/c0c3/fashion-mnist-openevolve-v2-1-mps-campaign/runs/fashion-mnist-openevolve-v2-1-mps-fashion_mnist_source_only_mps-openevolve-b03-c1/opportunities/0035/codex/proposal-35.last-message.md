MECHANISM: Final upper-edge EMA plateau bisection

HYPOTHESIS: A 28.61347198486328125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.

INTENDED_EDIT: Increase the endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2861347198486328125.

EVIDENCE: The current 0.2861328125 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.286136627197265625, loses one correct with lower cross-entropy; their midpoint is the most informative remaining boundary refinement.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.2861328125)
=======
                parameter.lerp_(shadow / correction, 0.2861347198486328125)
>>>>>>> REPLACE