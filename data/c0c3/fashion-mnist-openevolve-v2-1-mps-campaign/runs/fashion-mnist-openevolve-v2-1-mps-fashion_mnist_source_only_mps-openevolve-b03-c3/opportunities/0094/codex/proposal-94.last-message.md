MECHANISM: Axis-aware translation ensembling

HYPOTHESIS: Favoring horizontal over vertical radius-1 translations while retaining the best distance-stratified radius-2 weights will exceed 9,268 correct predictions by reducing harmful averaging across vertically informative garment positions.

INTENDED_EDIT: Preserve total TTA weight, increase each horizontal radius-1 shift from 0.09 to 0.10, decrease each vertical shift to 0.08, and restore the radius-2 weighting that tied the best correct count with lower cross-entropy.

EVIDENCE: Distance-stratified radius-2 TTA preserved 9,268 correct and improved cross-entropy to 0.2124557217; center/diagonal reallocations did not improve correctness, leaving axial orientation as an informative untested dimension.

<<<<<<< SEARCH
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
=======
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    if delta_y == 0:
                        weight = 0.10
                    elif delta_x == 0:
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
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE