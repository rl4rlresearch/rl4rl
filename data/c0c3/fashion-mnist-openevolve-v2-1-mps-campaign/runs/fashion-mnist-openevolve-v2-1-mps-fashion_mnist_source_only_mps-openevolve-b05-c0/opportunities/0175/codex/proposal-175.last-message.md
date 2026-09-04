MECHANISM: Boundary-refined terminal EMA interpolation

HYPOTHESIS: A 6.9873046875% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Increase only the forced final-iterate EMA interpolation from 6.984375% to 6.9873046875%, retaining regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Increasing terminal interpolation from 6.9375% to 6.984375% preserved 9,260 correct and reduced cross-entropy from 0.2120079094 to 0.2120078533; 6.9873046875% is the midpoint toward the timed-out 6.990234375% attempt, whose timeout supplied no contrary validation evidence.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06984375)
=======
                    average.lerp_(parameter.detach(), 0.069873046875)
>>>>>>> REPLACE