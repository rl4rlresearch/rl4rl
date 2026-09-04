MECHANISM: Fine-grained terminal-iterate EMA continuation

HYPOTHESIS: A 6.99609375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation from 6.984375% to 6.99609375%, retaining regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Raising terminal interpolation from 6.9375% to 6.984375% preserved 9,260 correct and lowered cross-entropy from 0.2120079094 to 0.2120078533; 6.99609375% is a conservative midpoint toward the timed-out 7.0078125% setting.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06984375)
=======
                    average.lerp_(parameter.detach(), 0.0699609375)
>>>>>>> REPLACE