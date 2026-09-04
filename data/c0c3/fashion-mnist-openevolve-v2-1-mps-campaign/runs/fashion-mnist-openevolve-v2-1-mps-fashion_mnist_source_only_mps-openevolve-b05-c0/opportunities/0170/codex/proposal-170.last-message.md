MECHANISM: Fine-grained terminal-iterate EMA interpolation

HYPOTHESIS: A 7.03125% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120079094.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation from 6.9375% to 7.03125%, retaining regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Raising terminal interpolation from 6.75% to 6.9375% preserved 9,260 correct and lowered cross-entropy from 0.2120081619 to 0.2120079094; 7.03125% is a conservative midpoint toward the unmeasured 7.125% setting.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.069375)
=======
                    average.lerp_(parameter.detach(), 0.0703125)
>>>>>>> REPLACE