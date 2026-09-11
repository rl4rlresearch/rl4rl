MECHANISM: Sign-aware vertical translation ensembling

HYPOTHESIS: Redistributing the successful vertical radius-1 weight from the downward-content view to the upward-content view will exceed 9,268 correct predictions by exploiting directional alignment while preserving the accuracy-safe total vertical weight.

INTENDED_EDIT: Restore the best distance-stratified radius-2 TTA and moderate vertical-over-horizontal weighting, then assign 0.11/0.09 weights to the two opposite vertical shifts instead of 0.10/0.10.

EVIDENCE: Moderate vertical bias retained 9,268 correct and improved cross-entropy, whereas stronger symmetric vertical bias lost five predictions; this motivates testing vertical directionality without increasing total vertical emphasis.

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
                    if delta_x == 0:
                        weight = 0.11 if delta_y > 0 else 0.09
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
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE