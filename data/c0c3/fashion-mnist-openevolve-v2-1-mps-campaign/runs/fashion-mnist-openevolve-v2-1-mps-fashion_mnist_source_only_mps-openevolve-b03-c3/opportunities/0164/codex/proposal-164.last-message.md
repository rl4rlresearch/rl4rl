MECHANISM: Continued center-compensated horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.069375 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions and reduce cross-entropy below 0.208131979.

INTENDED_EDIT: Advance one 0.0003125 step beyond Reference Design 2, transferring the combined 0.000625 horizontal-vote mass to the center while preserving all training behavior.

EVIDENCE: Five consecutive horizontal-weight reductions with fixed vertical influence retained 9,284 correct and monotonically lowered cross-entropy; Reference Design 2 reached the best result at 0.0696875.

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.11
                    elif delta_y == 0:
                        weight = 0.07
=======
        logits = 0.36375 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.069375
>>>>>>> REPLACE