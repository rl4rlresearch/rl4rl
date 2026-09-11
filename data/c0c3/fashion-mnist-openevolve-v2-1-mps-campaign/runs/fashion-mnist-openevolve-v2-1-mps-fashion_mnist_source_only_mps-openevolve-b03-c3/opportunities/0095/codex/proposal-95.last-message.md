MECHANISM: Vertical-axis-aware translation ensembling

HYPOTHESIS: Favoring vertical over horizontal radius-1 translations while restoring the best radius-2 weights will exceed 9,268 correct predictions because the opposite horizontal bias worsened cross-entropy.

INTENDED_EDIT: Increase each vertical radius-1 weight to 0.10, decrease each horizontal weight to 0.08, and restore the verified 0.009375/0.00625/0.003125 radius-2 weighting while preserving total ensemble weight.

EVIDENCE: Symmetric axis weights with mildly distance-stratified radius-2 TTA achieved 9,268 correct at 0.2124557 cross-entropy; horizontal bias retained 9,268 but worsened cross-entropy to 0.2124832, motivating the opposite directional test.

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.0125
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.0
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
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