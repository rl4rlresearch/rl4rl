MECHANISM: Continued center-compensated horizontal-vote pruning

HYPOTHESIS: Holding vertical weight at 0.10875 while reducing each horizontal radius-one weight to 0.07000 and transferring their combined mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132651.

INTENDED_EDIT: Advance one 0.0003125 step beyond Reference Design 2 along the validated horizontal-pruning direction, setting the center weight to 0.3625 while preserving total ensemble mass and all training behavior.

EVIDENCE: Three consecutive isolated horizontal-weight reductions retained 9,284 correct and monotonically improved cross-entropy from 0.208133718 to 0.208133353, 0.208132999, and 0.208132651; the regressions occurred when vertical influence was also increased.

<<<<<<< SEARCH
        logits = 0.360625 * self._flip_average(images)
=======
        logits = 0.3625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.0709375
=======
                    elif delta_y == 0:
                        weight = 0.07000
>>>>>>> REPLACE