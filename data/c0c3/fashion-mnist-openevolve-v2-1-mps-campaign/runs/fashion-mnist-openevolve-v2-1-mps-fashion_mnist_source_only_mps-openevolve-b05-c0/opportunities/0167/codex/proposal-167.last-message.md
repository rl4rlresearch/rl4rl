MECHANISM: Conservative final-iterate EMA continuation

HYPOTHESIS: A 6.75% terminal EMA interpolation will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120091377.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation from 6% to 6.75%, preserving regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct predictions and reduced cross-entropy from 0.2120130508 to 0.2120091377; 6.75% is a smaller continuation than the repeatedly timed-out 7.5% attempt.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06)
=======
                    average.lerp_(parameter.detach(), 0.0675)
>>>>>>> REPLACE