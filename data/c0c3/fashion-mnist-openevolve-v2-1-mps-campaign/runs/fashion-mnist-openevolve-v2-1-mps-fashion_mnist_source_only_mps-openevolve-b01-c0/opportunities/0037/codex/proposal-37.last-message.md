MECHANISM: Intermediate power-mean view ensemble

HYPOTHESIS: A power mean of order 0.5 will exceed 9,290 correct predictions by adding some consensus preference while retaining more of the arithmetic ensemble’s confident-view corrections than the geometric mean.

INTENDED_EDIT: Replace arithmetic probability averaging with a compute-neutral order-0.5 power mean across translated, flipped, live, and EMA predictions.

EVIDENCE: Geometric averaging improved cross-entropy from 0.22418 to 0.21799 but reduced correct predictions only slightly, from 9,290 to 9,286; an intermediate aggregation directly tests whether its calibration benefit can be retained without crossing the decision boundaries lost at the geometric endpoint.

<<<<<<< SEARCH
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        power = 0.5
        return (
            torch.logsumexp(
                power * torch.stack(log_probabilities, dim=0), dim=0
            )
            - math.log(len(log_probabilities))
        ) / power
>>>>>>> REPLACE