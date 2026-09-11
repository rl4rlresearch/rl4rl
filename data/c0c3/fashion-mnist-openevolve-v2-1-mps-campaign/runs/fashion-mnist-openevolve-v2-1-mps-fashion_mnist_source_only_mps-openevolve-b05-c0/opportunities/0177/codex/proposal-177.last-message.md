MECHANISM: Symmetric local terminal-EMA optimum probe

HYPOTHESIS: Reducing terminal EMA interpolation from 6.984375% to 6.9814453125% will preserve 9,260 correct predictions and lower cross-entropy below 0.2120078533.

INTENDED_EDIT: Test the point equally far below the best setting as the slightly worse 6.9873046875% result was above it, retaining all other behavior.

EVIDENCE: The 6.984375% setting achieved the best observed cross-entropy, while an upward change of 0.0029296875 percentage points regressed slightly with unchanged accuracy; the symmetric downward probe efficiently identifies whether the local optimum lies below the incumbent.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06984375)
=======
                    average.lerp_(parameter.detach(), 0.069814453125)
>>>>>>> REPLACE