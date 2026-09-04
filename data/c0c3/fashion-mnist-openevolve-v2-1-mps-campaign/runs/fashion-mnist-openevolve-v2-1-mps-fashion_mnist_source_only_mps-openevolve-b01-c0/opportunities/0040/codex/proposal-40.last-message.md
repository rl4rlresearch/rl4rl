MECHANISM: Confidence-emphasized power-mean view ensemble

HYPOTHESIS: An order-1.5 power mean will exceed 9,290 correct predictions by modestly emphasizing confident translated, flipped, live, and EMA predictions.

INTENDED_EDIT: Replace arithmetic probability averaging with a compute-neutral order-1.5 power mean.

EVIDENCE: Geometric averaging improved cross-entropy but reduced correct predictions from 9,290 to 9,286, indicating that suppressing confident-view corrections was harmful and motivating a controlled move beyond the arithmetic mean in the opposite direction.

<<<<<<< SEARCH
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        power = 1.5
        return (
            torch.logsumexp(
                power * torch.stack(log_probabilities, dim=0), dim=0
            )
            - math.log(len(log_probabilities))
        ) / power
>>>>>>> REPLACE