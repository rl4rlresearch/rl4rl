MECHANISM: Boundary-bisected cardinal-asymmetry TTA

HYPOTHESIS: Vertical/horizontal radius-one weights of 0.1090625/0.0709375 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.

INTENDED_EDIT: Bisect the interval between the best 0.10875/0.07125 setting and the 9,283-correct 0.109375/0.070625 setting, preserving total ensemble weight and all other behavior.

EVIDENCE: The 0.10875/0.07125 design achieved 9,284 correct with 0.208133718 cross-entropy, while 0.109375/0.070625 reduced cross-entropy to 0.208132253 but lost one correct prediction; their midpoint is the most direct probe of that decision boundary.

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.11
                    elif delta_y == 0:
                        weight = 0.07
=======
                    if delta_x == 0:
                        weight = 0.1090625
                    elif delta_y == 0:
                        weight = 0.0709375
>>>>>>> REPLACE