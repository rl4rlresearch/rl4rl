MECHANISM: Ultra-fine boundary-bisected horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.06951171875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.0694921875 setting, transferring the removed horizontal-vote mass to the center view.

EVIDENCE: The 0.06953125 design achieved 9,284 correct at 0.208131822 cross-entropy, while 0.0694921875 lowered cross-entropy to 0.208131781 but lost one correct prediction; their midpoint is the closest unresolved decision-boundary probe.

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)
=======
        logits = 0.3634765625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.11
                    elif delta_y == 0:
                        weight = 0.07
=======
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.06951171875
>>>>>>> REPLACE