MECHANISM: Decision-boundary bisection with center-compensated horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.069521484375 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.06951171875 setting, transferring the removed horizontal-vote mass to the center view.

EVIDENCE: The current 0.06953125 weight achieved 9,284 correct at 0.208131822 cross-entropy, while 0.06951171875 reduced cross-entropy to 0.208131799 but lost one correct prediction; their midpoint is the closest unresolved boundary probe.

<<<<<<< SEARCH
        logits = 0.3634375 * self._flip_average(images)
=======
        logits = 0.36345703125 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                        weight = 0.06953125
=======
                        weight = 0.069521484375
>>>>>>> REPLACE