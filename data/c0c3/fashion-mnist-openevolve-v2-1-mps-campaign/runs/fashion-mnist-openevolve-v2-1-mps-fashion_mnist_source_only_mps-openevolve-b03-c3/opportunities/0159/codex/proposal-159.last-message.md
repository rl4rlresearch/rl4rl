MECHANISM: Further center-compensated horizontal-vote pruning

HYPOTHESIS: Keeping vertical weight at 0.10875 while reducing horizontal weight to 0.070625 and transferring the removed mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208133353.

INTENDED_EDIT: Apply the next 0.0003125 horizontal-weight reduction beyond Reference Design 3, increasing the center weight by 0.000625 to preserve total ensemble mass.

EVIDENCE: Reference Design 3 retained 9,284 correct and improved cross-entropy to 0.208133353 by reducing horizontal influence while holding vertical influence fixed; this continues that isolated direction without the vertical increase associated with prior one-prediction regressions.

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
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
=======
        logits = 0.36125 * self._flip_average(images)

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
                        weight = 0.070625
                    else:
                        weight = 0.045
>>>>>>> REPLACE