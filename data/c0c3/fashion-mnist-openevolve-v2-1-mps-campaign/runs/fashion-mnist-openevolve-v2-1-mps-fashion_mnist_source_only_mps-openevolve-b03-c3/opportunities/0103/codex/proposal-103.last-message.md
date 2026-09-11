MECHANISM: Multiscale vertical-axis translation ensembling

HYPOTHESIS: Extending the successful moderate vertical preference to radius-2 TTA views will exceed 9,268 correct predictions while preserving total ensemble weight.

INTENDED_EDIT: Restore Reference Design 3’s accuracy-safe radius-1 weights and distance-stratified radius-2 weights, then mildly favor vertically dominant radius-2 translations over horizontally dominant ones.

EVIDENCE: Vertical-biased radius-1 TTA achieved the best verified tied score at 9,268 correct and 0.2124414 cross-entropy, while horizontal bias worsened cross-entropy and stronger radius-1 bias lost accuracy; radius-2 orientation remains untested.

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
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
                        weight = (
                            0.010625 if delta_x == 0 else 0.008125
                        )
                    elif manhattan_distance == 3:
                        weight = (
                            0.006875
                            if abs(delta_y) > abs(delta_x)
                            else 0.005625
                        )
                    else:
                        weight = 0.003125
>>>>>>> REPLACE