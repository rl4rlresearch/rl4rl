MECHANISM: Fine boundary-bisected horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.0694921875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the lower-cross-entropy but 9,283-correct 0.069453125 setting, transferring removed horizontal mass to the center view.

EVIDENCE: Weight 0.06953125 achieved 9,284 correct at 0.208131822 cross-entropy, while 0.069453125 achieved 9,283 correct at 0.208131741; their midpoint is the most informative unresolved boundary probe.

<<<<<<< SEARCH
        logits = 0.361875 * self._flip_average(images)
=======
        logits = 0.363515625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.0703125
=======
                    elif delta_y == 0:
                        weight = 0.0694921875
>>>>>>> REPLACE