MECHANISM: Center-compensated horizontal-vote pruning

HYPOTHESIS: Reducing radius-one horizontal weight to 0.0709375 while transferring its total mass to the center view will retain at least 9,284 correct predictions and lower cross-entropy below 0.208133718.

INTENDED_EDIT: Restore the best validated 0.10875 vertical weight, reduce horizontal weight without increasing vertical influence, and raise the center weight to preserve total ensemble mass.

EVIDENCE: The 0.10875/0.07125 design achieved 9,284 correct, while the coupled 0.1090625/0.0709375 change lost one; isolating the horizontal reduction tests whether the loss came from increased vertical influence while favoring the reliable center prediction.

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)
=======
        logits = 0.360625 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.105
                    elif delta_y == 0:
                        weight = 0.075
=======
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.0709375
>>>>>>> REPLACE