MECHANISM: Stronger sparse tail interpolation

HYPOTHESIS: Blending 75% toward the sparse tail mean will retain at least 9,286 correct predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Increase the final parameter interpolation coefficient from 0.5 to 0.75.

EVIDENCE: Half-strength averaging achieved 9,286 correct, while full averaging lost only one prediction but improved cross-entropy to 0.19631; testing the midpoint between them targets better calibration with less boundary drift than full averaging.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.75)
>>>>>>> REPLACE