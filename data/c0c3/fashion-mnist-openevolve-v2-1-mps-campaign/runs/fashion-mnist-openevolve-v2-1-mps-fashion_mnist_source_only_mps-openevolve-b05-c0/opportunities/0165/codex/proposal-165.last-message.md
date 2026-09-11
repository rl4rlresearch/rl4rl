MECHANISM: Intermediate final-iterate EMA interpolation

HYPOTHESIS: A 7.5% terminal EMA interpolation will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.

INTENDED_EDIT: Increase only the forced final-iterate EMA interpolation from 6% to 7.5%; retain 3% quarter-rate EMA updates and all other training behavior.

EVIDENCE: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct and reduced cross-entropy from 0.2120130508 to 0.2120091377. The 9% attempt timed out without contrary validation evidence, so 7.5% is a conservative intermediate continuation.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.06)
=======
                    average.lerp_(parameter.detach(), 0.075)
>>>>>>> REPLACE