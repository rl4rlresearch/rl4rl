MECHANISM: Geometric-mean view ensemble

HYPOTHESIS: Averaging log-probabilities instead of probabilities will exceed 9,290 correct predictions by favoring class consensus across translated, flipped, live, and EMA views rather than allowing a single overconfident view to dominate.

INTENDED_EDIT: Replace the arithmetic probability mixture with an equal-weight geometric probability mixture, requiring no additional forward passes or training changes.

EVIDENCE: Translation-flip ensembling previously improved correct predictions from 9,138 to 9,167, while changing live/EMA weights in either direction regressed from 9,290; this motivates preserving equal weights while testing the previously unexamined aggregation rule.

<<<<<<< SEARCH
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        return torch.stack(log_probabilities, dim=0).mean(dim=0)
>>>>>>> REPLACE