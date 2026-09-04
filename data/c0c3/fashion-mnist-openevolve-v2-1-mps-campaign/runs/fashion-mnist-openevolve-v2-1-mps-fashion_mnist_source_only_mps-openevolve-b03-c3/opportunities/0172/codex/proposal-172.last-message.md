MECHANISM: Center-compensated vertical-vote pruning

HYPOTHESIS: With the best validated horizontal weight fixed at 0.06953125, reducing each radius-one vertical weight to 0.1084375 and transferring their combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.

INTENDED_EDIT: Restore Reference Design 2’s best horizontal weighting, then take one isolated 0.0003125 downward step in each vertical vote and add the removed 0.000625 total weight to the center view.

EVIDENCE: Horizontal pruning has reached a sharp 9,284/9,283 decision boundary, while increased vertical influence was associated with regressions; an isolated downward vertical step is the closest untested, evidence-backed TTA direction.

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
        logits = 0.3640625 * self._flip_average(images)

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
                        weight = 0.1084375
                    elif delta_y == 0:
                        weight = 0.06953125
>>>>>>> REPLACE