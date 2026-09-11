MECHANISM: Midpoint terminal-iterate EMA refinement

HYPOTHESIS: A 6.9375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120081619.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation from 6.75% to 6.9375%, retaining regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Increasing terminal interpolation from 6% to 6.75% preserved 9,260 correct and lowered cross-entropy from 0.2120091377 to 0.2120081619; 6.9375% is the midpoint toward the 7.125% attempt, whose timeout provided no contrary accuracy evidence.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.0675)
=======
                    average.lerp_(parameter.detach(), 0.069375)
>>>>>>> REPLACE