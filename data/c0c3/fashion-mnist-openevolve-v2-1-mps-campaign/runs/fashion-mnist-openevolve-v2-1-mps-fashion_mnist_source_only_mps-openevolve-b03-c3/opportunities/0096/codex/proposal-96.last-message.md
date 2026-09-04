MECHANISM: Stronger vertical-axis translation ensembling

HYPOTHESIS: Increasing the vertical radius-1 TTA weights to 0.11 and decreasing the horizontal weights to 0.07 will exceed 9,268 correct predictions by extending the cross-entropy improvement observed when moving from horizontal bias to vertical bias.

INTENDED_EDIT: Restore the best 0.36 center and 0.045 diagonal weights, then strengthen the verified vertical-over-horizontal radius-1 weighting while preserving total ensemble weight.

EVIDENCE: Reference Design 3’s 0.10 vertical/0.08 horizontal weighting retained 9,268 correct and achieved the best cross-entropy, 0.2124414, while the opposite bias worsened cross-entropy to 0.2124832; this tests whether that directional trend continues.

<<<<<<< SEARCH
        logits = 0.38 * self._flip_average(images)

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
                        else 0.04
                    )
=======
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
                    else:
                        weight = 0.045
>>>>>>> REPLACE