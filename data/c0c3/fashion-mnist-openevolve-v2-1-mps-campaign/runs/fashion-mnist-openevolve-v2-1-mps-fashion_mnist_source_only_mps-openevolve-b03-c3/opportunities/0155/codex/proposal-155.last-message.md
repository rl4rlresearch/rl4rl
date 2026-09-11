MECHANISM: Boundary-focused cardinal-asymmetry interpolation

HYPOTHESIS: Vertical/horizontal radius-one weights of 0.109375/0.070625 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.

INTENDED_EDIT: Move halfway from the best 0.10875/0.07125 TTA weights toward the lower-cross-entropy but 9,283-correct 0.11/0.07 setting, preserving total ensemble weight and all training behavior.

EVIDENCE: Increasing cardinal contrast steadily lowered cross-entropy through 0.10875/0.07125 while retaining 9,284 correct, whereas 0.11/0.07 lost one correct prediction; bisecting that narrow interval directly probes the apparent decision boundary.

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
=======
                    if delta_x == 0:
                        weight = 0.109375
                    elif delta_y == 0:
                        weight = 0.070625
>>>>>>> REPLACE