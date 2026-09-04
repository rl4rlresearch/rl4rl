MECHANISM: Amplified cardinal-direction TTA contrast

HYPOTHESIS: Increasing the validated vertical-over-horizontal radius-one weighting contrast from 0.02 to 0.04 while preserving total ensemble weight will exceed 9,284 correct predictions.

INTENDED_EDIT: Weight vertical radius-one shifts at 0.11 and horizontal shifts at 0.07; retain diagonal, radius-two, architecture, and training settings.

EVIDENCE: The 0.10/0.08 asymmetric design achieved 9,284 correct versus 9,283 for equal 0.09 weights, while flattening radius-two weights regressed, motivating a focused extrapolation of the beneficial cardinal asymmetry.

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0 or delta_y == 0:
                        weight = 0.09
                    else:
                        weight = 0.045
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.11
                    elif delta_y == 0:
                        weight = 0.07
                    else:
                        weight = 0.045
>>>>>>> REPLACE