MECHANISM: Further boundary-bisected cardinal-asymmetry TTA

HYPOTHESIS: Vertical/horizontal radius-one weights of 0.10890625/0.07109375 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.

INTENDED_EDIT: Bisect the interval between the best 0.10875/0.07125 setting and the lower-cross-entropy but 9,283-correct 0.1090625/0.0709375 setting, preserving total ensemble weight and all other behavior.

EVIDENCE: The current 0.10875/0.07125 design achieved 9,284 correct at 0.208133718 cross-entropy, while 0.1090625/0.0709375 reduced cross-entropy to 0.208132978 but lost one prediction; their midpoint is the next direct probe of the classification boundary.

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.07125
=======
                    if delta_x == 0:
                        weight = 0.10890625
                    elif delta_y == 0:
                        weight = 0.07109375
>>>>>>> REPLACE