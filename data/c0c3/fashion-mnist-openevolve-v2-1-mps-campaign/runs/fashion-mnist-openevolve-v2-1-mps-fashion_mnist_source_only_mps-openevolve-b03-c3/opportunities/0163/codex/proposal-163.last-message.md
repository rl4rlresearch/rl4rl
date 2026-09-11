MECHANISM: Continued center-compensated horizontal-vote pruning

HYPOTHESIS: Reducing each radius-one horizontal weight to 0.0696875 and transferring the combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208132310.

INTENDED_EDIT: Advance one 0.0003125 step beyond Reference Design 3 along the validated horizontal-pruning direction, increasing the center weight to 0.363125 while preserving total ensemble mass and all training behavior.

EVIDENCE: Four consecutive center-compensated horizontal reductions retained 9,284 correct while monotonically lowering cross-entropy, most recently from 0.208132651 at weight 0.0703125 to 0.208132310 at 0.07000.

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)
=======
        logits = 0.363125 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.07125
=======
                    elif delta_y == 0:
                        weight = 0.0696875
>>>>>>> REPLACE