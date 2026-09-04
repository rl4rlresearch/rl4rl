MECHANISM: Continued center-compensated horizontal-vote pruning

HYPOTHESIS: Reducing radius-one horizontal weight from 0.070625 to 0.0703125 while transferring its total mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132999.

INTENDED_EDIT: Decrease each horizontal radius-one TTA vote by 0.0003125 and increase the center vote by 0.000625, preserving total ensemble weight and all training behavior.

EVIDENCE: Two consecutive center-compensated horizontal reductions retained 9,284 correct while improving cross-entropy from 0.208133718 to 0.208133353 and then 0.208132999; continuing the same isolated direction is the closest evidence-backed probe.

<<<<<<< SEARCH
        logits = 0.36125 * self._flip_average(images)
=======
        logits = 0.361875 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.070625
=======
                    elif delta_y == 0:
                        weight = 0.0703125
>>>>>>> REPLACE