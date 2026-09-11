MECHANISM: Local terminal-EMA optimum refinement

HYPOTHESIS: A 6.98583984375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Test the midpoint between the best 6.984375% interpolation and the slightly worse 6.9873046875% setting, retaining all other training behavior.

EVIDENCE: The 6.984375% setting achieved the best observed cross-entropy of 0.2120078533, while 6.9873046875% regressed slightly to 0.2120078560 with unchanged accuracy; their midpoint is the most informative local refinement.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06984375)
=======
                    average.lerp_(parameter.detach(), 0.0698583984375)
>>>>>>> REPLACE