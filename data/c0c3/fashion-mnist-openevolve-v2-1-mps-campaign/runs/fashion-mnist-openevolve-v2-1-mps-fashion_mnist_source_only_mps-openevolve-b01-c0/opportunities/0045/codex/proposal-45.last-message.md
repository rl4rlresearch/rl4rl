MECHANISM: Decision-preserving post-ensemble temperature sharpening

HYPOTHESIS: Scaling the arithmetic ensemble logits by 1/0.9 will retain the baseline’s 9,290 correct predictions while reducing validation cross-entropy below 0.2241766, thereby improving validation_score.

INTENDED_EDIT: Apply temperature 0.9 to the final live/EMA multi-view ensemble without changing training, parameters, forward-pass count, or predicted classes.

EVIDENCE: Geometric aggregation reduced cross-entropy to 0.217986 but lost four correct predictions; positive temperature scaling can pursue that calibration improvement while preserving the arithmetic ensemble’s argmax decisions.

<<<<<<< SEARCH
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return ensemble_log_probabilities / 0.9
>>>>>>> REPLACE