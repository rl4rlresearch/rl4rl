MECHANISM: Center-biased inner-translation ensembling

HYPOTHESIS: Restoring batch size 192 and shifting 0.02 ensemble weight from radius-1 diagonal views to the unshifted view will exceed 9,268 correct predictions by reducing reliance on more-displaced inputs while preserving total calibration weight.

INTENDED_EDIT: Restore the best verified batch size, increase the unshifted TTA weight from 0.36 to 0.38, and reduce each radius-1 diagonal weight from 0.045 to 0.040.

EVIDENCE: Batch size 192 achieved 9,268 correct versus 9,263 at 128 and 9,243 at 160; mildly favoring less-displaced radius-2 views preserved all 9,268 predictions and improved cross-entropy, motivating the same conservative distance bias among inner views.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
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
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
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
>>>>>>> REPLACE