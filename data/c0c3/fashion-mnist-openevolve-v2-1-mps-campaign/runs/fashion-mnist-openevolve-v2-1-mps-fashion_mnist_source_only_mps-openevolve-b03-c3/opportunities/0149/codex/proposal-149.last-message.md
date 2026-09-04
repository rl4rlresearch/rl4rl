MECHANISM: Decision-boundary cardinal-asymmetry refinement

HYPOTHESIS: Vertical/horizontal radius-one weights of 0.10875/0.07125 will retain at least 9,284 correct predictions while lowering cross-entropy below 0.208136808, improving validation_score.

INTENDED_EDIT: Move halfway from the current 0.1075/0.0725 TTA weights toward the lower-cross-entropy 0.11/0.07 setting, preserving total ensemble weight and all training behavior.

EVIDENCE: Cross-entropy improved monotonically as cardinal contrast increased from 0.10/0.08 through 0.105/0.075 to 0.1075/0.0725, all retaining 9,284 correct, while 0.11/0.07 lowered cross-entropy further but lost one prediction; this tests the midpoint nearest that apparent classification boundary.

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.1075
                    elif delta_y == 0:
                        weight = 0.0725
=======
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.07125
>>>>>>> REPLACE