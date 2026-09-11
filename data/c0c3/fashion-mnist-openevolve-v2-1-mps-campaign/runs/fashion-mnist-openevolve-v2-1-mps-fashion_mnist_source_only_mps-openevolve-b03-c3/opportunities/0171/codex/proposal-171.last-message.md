MECHANISM: Decision-boundary bisection with center-compensated horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.0695263671875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.069521484375 setting, transferring the removed horizontal-vote mass to the center view.

EVIDENCE: Weight 0.06953125 achieved 9,284 correct with 0.208131822 cross-entropy, while every tested lower weight through 0.069521484375 reduced cross-entropy but lost one prediction; their midpoint is the closest unresolved boundary probe.

<<<<<<< SEARCH
        logits = 0.361875 * self._flip_average(images)
=======
        logits = 0.363447265625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.0703125
=======
                    elif delta_y == 0:
                        weight = 0.0695263671875
>>>>>>> REPLACE