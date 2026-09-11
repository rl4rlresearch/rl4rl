MECHANISM: Conservative decision-preserving temperature sharpening

HYPOTHESIS: Applying temperature 0.95 to the arithmetic ensemble will preserve its 9,290 correct predictions while lowering cross-entropy below 0.2241766, producing a strict validation_score improvement.

INTENDED_EDIT: Scale the final ensemble log-probabilities by 1/0.95 without changing training, parameters, forward-pass count, or predicted classes.

EVIDENCE: Geometric aggregation lowered cross-entropy from 0.2241766 to 0.217986, suggesting the baseline ensemble benefits from sharper consensus; the prior temperature-0.9 attempt timed out and therefore did not test this mechanism.

<<<<<<< SEARCH
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return ensemble_log_probabilities / 0.95
>>>>>>> REPLACE