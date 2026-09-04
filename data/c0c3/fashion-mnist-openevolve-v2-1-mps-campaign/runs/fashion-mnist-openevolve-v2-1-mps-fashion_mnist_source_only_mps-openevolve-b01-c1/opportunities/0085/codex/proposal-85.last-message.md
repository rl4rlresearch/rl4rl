MECHANISM: Arithmetic probability test-time aggregation

HYPOTHESIS: Averaging per-view class probabilities will exceed 9,247 correct predictions by preventing a single overconfident crop or flip from disproportionately steering the ensemble.

INTENDED_EDIT: Replace validation-time logit averaging with weighted probability averaging, then convert the resulting distribution back to temperature-scaled logits.

EVIDENCE: Changes to crop weights and supervision schedules repeatedly failed to improve the 9,247-correct design, while the fundamental aggregation rule remains untested; this isolates that rule without altering learned parameters or training.

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
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 18.0
        return 1.29834 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        probabilities_sum = None
        central_probabilities_sum = None
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
                if probabilities_sum is None:
                    probabilities_sum = view_probabilities
                else:
                    probabilities_sum = (
                        probabilities_sum + view_probabilities
                    )
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_probabilities_sum is None:
                        central_probabilities_sum = view_probabilities
                    else:
                        central_probabilities_sum = (
                            central_probabilities_sum + view_probabilities
                        )
        full_ensemble = probabilities_sum / 50.0
        central_ensemble = central_probabilities_sum / 18.0
        ensemble_probabilities = (
            0.9 * full_ensemble + 0.1 * central_ensemble
        )
        return 1.29834 * ensemble_probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE