MECHANISM: Arithmetic–geometric ensemble interpolation

HYPOTHESIS: Blending 25% mean log-probability evidence into the verified probability ensemble will retain at least 9,252 correct predictions while lowering cross-entropy below 0.2235582.

INTENDED_EDIT: Interpolate between probability averaging and the recently tested logit-equivalent geometric aggregation, retaining all ten views and the verified 1.10 calibration.

EVIDENCE: Direct logit averaging improved cross-entropy from 0.2235582 to 0.2198113 while losing only one correct prediction; a conservative interpolation should capture some calibration benefit without crossing that decision boundary.

<<<<<<< SEARCH
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return 1.10 * ensemble_log_probabilities
=======
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities, dim=0
        ) - math.log(len(log_probabilities))
        geometric_ensemble = stacked_log_probabilities.mean(dim=0)
        ensemble_scores = (
            0.75 * probability_ensemble + 0.25 * geometric_ensemble
        )
        return 1.10 * ensemble_scores
>>>>>>> REPLACE