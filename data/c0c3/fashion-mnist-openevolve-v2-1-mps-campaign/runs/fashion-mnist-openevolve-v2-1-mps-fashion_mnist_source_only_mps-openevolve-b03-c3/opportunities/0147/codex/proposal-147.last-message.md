MECHANISM: Midpoint cardinal-asymmetry TTA

HYPOTHESIS: Vertical/horizontal radius-one weights of 0.105/0.075 with the validated distance-sensitive outer weights will exceed 9,284 correct predictions by refining the apparent optimum between the 0.10/0.08 and 0.11/0.07 designs.

INTENDED_EDIT: Restore distance-dependent radius-two weighting and test the midpoint between the best 0.10/0.08 cardinal contrast and the regressed 0.11/0.07 contrast, preserving total ensemble weight.

EVIDENCE: The 0.10/0.08 design achieved the best 9,284 correct, while both 0.09/0.09 and 0.11/0.07 achieved 9,283; uniform radius-two weights also regressed to 9,282, motivating a narrow cardinal-weight search with the validated outer weighting restored.

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0 or delta_y == 0:
                        weight = 0.09
                    else:
                        weight = 0.045
                else:
                    weight = 0.00625
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.105
                    elif delta_y == 0:
                        weight = 0.075
                    else:
                        weight = 0.045
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
>>>>>>> REPLACE