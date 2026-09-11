MECHANISM: Slight EMA-favoring probability ensemble

HYPOTHESIS: Weighting the smoothed EMA predictions at 51% and the endpoint model at 49% will either exceed 9,290 correct predictions or preserve them while reducing cross-entropy below 0.20249243125915528.

INTENDED_EDIT: Replace equal live/EMA probability averaging with a normalized 49:51 mixture while retaining the best verified temperature.

EVIDENCE: Both adjacent float32 temperatures were worse than the current value, exhausting local scalar calibration; ensemble composition is the next orthogonal refinement, and the EMA represents a smoothed model estimate rather than a single training endpoint.

<<<<<<< SEARCH
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return ensemble_log_probabilities / 0.717143714427948
=======
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        if ema_state is None:
            ensemble_log_probabilities = torch.logsumexp(
                stacked_log_probabilities, dim=0
            ) - math.log(len(log_probabilities))
        else:
            view_count = len(log_probabilities) // 2
            log_weights = stacked_log_probabilities.new_empty(
                (len(log_probabilities), 1, 1)
            )
            log_weights[:view_count].fill_(math.log(0.49 / view_count))
            log_weights[view_count:].fill_(math.log(0.51 / view_count))
            ensemble_log_probabilities = torch.logsumexp(
                stacked_log_probabilities + log_weights, dim=0
            )
        return ensemble_log_probabilities / 0.717143714427948
>>>>>>> REPLACE