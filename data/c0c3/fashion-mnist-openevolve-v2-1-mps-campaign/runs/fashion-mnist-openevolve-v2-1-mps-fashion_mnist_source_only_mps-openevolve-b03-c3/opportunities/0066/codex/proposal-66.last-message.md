MECHANISM: Posterior-mixture shift ensembling

HYPOTHESIS: Averaging per-shift class probabilities instead of logits will exceed 9,240 correct predictions by limiting the influence of overconfident errors from individual translated views.

INTENDED_EDIT: Adopt the best verified 0.04→0 smoothing schedule and 1.285 calibration, while changing only the 25-shift aggregation from a weighted logit average to a weighted posterior mixture; flip pairs remain logit-averaged.

EVIDENCE: Reference Design 1 achieved the best verified score with 9,240 correct using 0.04→0 smoothing and 1.285 calibration. Added spatial refinement and phase-stable pooling reduced correctness, while the design’s extensive shift evaluation indicates residual view sensitivity, motivating a robust aggregation change without altering learned capacity.

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
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
        return 1.296352 * logits
=======
        probabilities = 0.36 * self._flip_average(images).softmax(dim=1)

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
                    weight = 0.00625
                shifted_probabilities = self._flip_average(shifted).softmax(
                    dim=1
                )
                probabilities = probabilities + weight * shifted_probabilities
        return 1.285 * probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE