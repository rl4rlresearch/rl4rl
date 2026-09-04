MECHANISM: Diagonal-favoring inner-translation ensembling

HYPOTHESIS: Reversing the unsuccessful center bias by moving 0.02 ensemble weight from the unshifted view to radius-1 diagonal views will exceed 9,268 correct predictions.

INTENDED_EDIT: Reduce the unshifted TTA weight from 0.36 to 0.34 and increase each radius-1 diagonal weight from 0.045 to 0.050, preserving total ensemble weight and all training settings.

EVIDENCE: Moving the same 0.02 weight in the opposite direction—from diagonal views to the center—reduced correctness from 9,268 to 9,265, indicating that radius-1 diagonal evidence is more valuable than additional center weight.

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
=======
        logits = 0.34 * self._flip_average(images)

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
                        else 0.050
                    )
>>>>>>> REPLACE