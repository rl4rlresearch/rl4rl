MECHANISM: Conservative terminal EMA interpolation refinement

HYPOTHESIS: A 7.125% terminal EMA interpolation will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120081619.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation from 6.75% to 7.125%, preserving regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Terminal interpolation increases from 3% to 6% to 6.75% successively lowered cross-entropy from 0.2120130508 to 0.2120091377 to 0.2120081619 without changing the 9,260 correct predictions; 7.125% is a conservative midpoint toward the repeatedly timed-out 7.5% test.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.0675)
=======
                    average.lerp_(parameter.detach(), 0.07125)
>>>>>>> REPLACE