MECHANISM: Boundary-bisected center-compensated horizontal-vote pruning

HYPOTHESIS: Horizontal weight 0.06953125 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while reducing cross-entropy below 0.208131979.

INTENDED_EDIT: Bisect the interval between the best 0.0696875 horizontal weight and the 9,283-correct 0.069375 setting, transferring the removed horizontal mass to the center view.

EVIDENCE: Weight 0.0696875 achieved 9,284 correct at 0.208131979 cross-entropy, while 0.069375 lowered cross-entropy but lost one correct prediction; their midpoint is the most informative probe of the decision boundary.

<<<<<<< SEARCH
        logits = 0.363125 * self._flip_average(images)

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
                        weight = 0.0696875
=======
        logits = 0.3634375 * self._flip_average(images)

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
                        weight = 0.06953125
>>>>>>> REPLACE