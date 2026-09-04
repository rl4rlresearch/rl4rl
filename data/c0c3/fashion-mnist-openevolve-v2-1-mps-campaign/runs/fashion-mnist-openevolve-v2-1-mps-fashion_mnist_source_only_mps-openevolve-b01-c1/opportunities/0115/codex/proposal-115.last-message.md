MECHANISM: Probability-space test-time augmentation pooling

HYPOTHESIS: Averaging normalized class probabilities across translated and flipped views will exceed 9,257 correct predictions by preventing an overconfident misaligned crop from dominating the ensemble.

INTENDED_EDIT: Replace validation-time arithmetic logit averaging with probability averaging, then return the log-probability mixture with the existing calibration scale.

EVIDENCE: Center-weighted crop aggregation improved the best design, showing that predictions vary meaningfully across geometric views; probability pooling directly targets that variation without changing the proven training procedure or parameter count.

<<<<<<< SEARCH
        logits_sum = None
        central_logits_sum = None
        for offset_y in range(5):
            for offset_x in range(5):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    central_weight = (
                        (2 if offset_y == 2 else 1)
                        * (2 if offset_x == 2 else 1)
                    )
                    if central_logits_sum is None:
                        central_logits_sum = central_weight * view_logits
                    else:
                        central_logits_sum = (
                            central_logits_sum
                            + central_weight * view_logits
                        )
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 32.0
        return 1.29834 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        probability_sum = None
        central_probability_sum = None
        for offset_y in range(5):
            for offset_x in range(5):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                probabilities = self._forward_once(views).softmax(dim=-1)
                original_probabilities, flipped_probabilities = (
                    probabilities.chunk(2, dim=0)
                )
                view_probabilities = (
                    original_probabilities + flipped_probabilities
                )
                if probability_sum is None:
                    probability_sum = view_probabilities
                else:
                    probability_sum = probability_sum + view_probabilities
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    central_weight = (
                        (2 if offset_y == 2 else 1)
                        * (2 if offset_x == 2 else 1)
                    )
                    if central_probability_sum is None:
                        central_probability_sum = (
                            central_weight * view_probabilities
                        )
                    else:
                        central_probability_sum = (
                            central_probability_sum
                            + central_weight * view_probabilities
                        )
        full_ensemble = probability_sum / 50.0
        central_ensemble = central_probability_sum / 32.0
        ensemble_probabilities = (
            0.9 * full_ensemble + 0.1 * central_ensemble
        )
        return 1.29834 * ensemble_probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE