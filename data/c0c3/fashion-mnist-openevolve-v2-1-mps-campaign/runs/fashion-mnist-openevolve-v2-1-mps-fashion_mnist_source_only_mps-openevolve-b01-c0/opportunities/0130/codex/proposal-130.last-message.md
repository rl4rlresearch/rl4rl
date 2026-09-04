MECHANISM: Canonical-view-weighted test-time augmentation

HYPOTHESIS: Increasing the unshifted view’s ensemble weight from 20% to 27.27% will correct at least one shift-induced prediction error and exceed 9,290 validation-correct predictions.

INTENDED_EDIT: Weight the original and horizontally flipped original views 1.5× while retaining all shifted views, the live/EMA mixture, and temperature calibration.

EVIDENCE: Adjacent live/EMA weights and temperatures only tied or worsened the 9,290-correct result, indicating those calibration coordinates are saturated; view weighting is an independent prediction coordinate, and the unshifted views preserve the validation image geometry without synthetic translation.

<<<<<<< SEARCH
        live_log_probabilities = collect_log_probabilities()
        live_ensemble = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0), dim=0
        ) - math.log(len(live_log_probabilities))
=======
        live_log_probabilities = collect_log_probabilities()
        view_weights = live_log_probabilities[0].new_tensor(
            (1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        )
        log_view_weights = view_weights.log().view(-1, 1, 1)
        log_view_weight_sum = view_weights.sum().log()
        live_ensemble = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0) + log_view_weights,
            dim=0,
        ) - log_view_weight_sum
>>>>>>> REPLACE

<<<<<<< SEARCH
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(len(ema_log_probabilities))
=======
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0) + log_view_weights,
                dim=0,
            ) - log_view_weight_sum
>>>>>>> REPLACE