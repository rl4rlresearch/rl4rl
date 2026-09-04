MECHANISM: Three-point parabolic terminal-EMA refinement

HYPOTHESIS: A 6.9847005% terminal EMA interpolation will preserve 9,260 correct predictions and reduce cross-entropy below 0.2120078533.

INTENDED_EDIT: Replace only the terminal optimizer-iterate EMA interpolation with the quadratic minimum estimated from the incumbent and the two symmetric probes.

EVIDENCE: The 6.984375% incumbent outperformed both equidistant probes: 6.9814453125% increased cross-entropy by 4.20e-9, while 6.9873046875% increased it by 2.67e-9. Their asymmetric regressions estimate the local minimum slightly above the incumbent.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06984375)
=======
                    average.lerp_(parameter.detach(), 0.069847005)
>>>>>>> REPLACE