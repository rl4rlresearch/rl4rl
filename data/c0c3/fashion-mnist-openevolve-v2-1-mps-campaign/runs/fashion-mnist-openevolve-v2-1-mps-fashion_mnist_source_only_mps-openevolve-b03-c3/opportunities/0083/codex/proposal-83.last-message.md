MECHANISM: Probability-space translation ensembling

HYPOTHESIS: Averaging translation predictions in probability space with the best mild distance-stratified weights will exceed 9,268 correct predictions by limiting the influence of overconfident errors from displaced views.

INTENDED_EDIT: Restore the best radius-2 weights and replace translation-logit averaging with calibrated probability averaging; training, flip-logit averaging, architecture, and parameter count remain unchanged.

EVIDENCE: Mild distance-stratified TTA retained 9,268 correct and improved cross-entropy, while stronger axial weighting lost one prediction. This suggests retaining all translated evidence but making aggregation more robust to individual harmful views.

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
                logits = logits + weight * self._flip_average(shifted)
        return 1.253 * logits
=======
        probabilities = 0.36 * F.softmax(
            self._flip_average(images), dim=1
        )

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
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                probabilities = probabilities + weight * F.softmax(
                    self._flip_average(shifted), dim=1
                )
        return 1.253 * probabilities.clamp_min(1.0e-12).log()
>>>>>>> REPLACE