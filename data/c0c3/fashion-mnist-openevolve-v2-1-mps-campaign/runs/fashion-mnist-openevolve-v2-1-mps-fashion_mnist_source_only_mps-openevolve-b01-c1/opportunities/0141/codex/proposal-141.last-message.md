MECHANISM: Confidence-normalized test-time augmentation pooling

HYPOTHESIS: Arithmetic probability pooling across shifted and flipped views will exceed 9,265 correct predictions by preventing a single overconfident crop from dominating the ensemble.

INTENDED_EDIT: Softmax-normalize each augmentation prediction before averaging, then return scaled log-probabilities as ten-class logits.

EVIDENCE: The 233,434-parameter architecture with the 5% learning-rate floor remains best, while capacity changes, EMA, and attention failed or regressed; this preserves that training trajectory and tests the previously unchanged aggregation of its existing 50 validation views.

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
                logits = self._forward_once(views)
                probabilities = F.softmax(logits, dim=-1)
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
                view_probabilities = original_probs + flipped_probs
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
        ensemble = 0.9 * full_ensemble + 0.1 * central_ensemble
        return 1.29834 * ensemble.clamp_min(1.0e-7).log()
>>>>>>> REPLACE