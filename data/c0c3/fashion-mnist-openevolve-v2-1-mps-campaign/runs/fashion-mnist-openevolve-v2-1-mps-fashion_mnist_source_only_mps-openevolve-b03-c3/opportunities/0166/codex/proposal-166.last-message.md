MECHANISM: Boundary-bisected center-compensated horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.069453125 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Move halfway from the best validated 0.06953125 horizontal weight toward the lower-cross-entropy but 9,283-correct 0.069375 setting, transferring the removed horizontal mass to the center view.

EVIDENCE: Weight 0.06953125 achieved 9,284 correct at 0.208131822 cross-entropy, whereas 0.069375 reduced cross-entropy to 0.208131659 but lost one correct prediction; bisecting this interval is the closest evidence-backed probe of the boundary.

<<<<<<< SEARCH
        logits = 0.3625 * self._flip_average(images)
=======
        logits = 0.36359375 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    elif delta_y == 0:
                        weight = 0.07000
=======
                    elif delta_y == 0:
                        weight = 0.069453125
>>>>>>> REPLACE