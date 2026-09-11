MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.09 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2272195.

INTENDED_EDIT: Apply a 1.09 multiplier to the aggregate inference logits while leaving training and ensemble weighting unchanged.

EVIDENCE: Increasing the multiplier from 1.03 to 1.06 preserved 9,256 correct predictions and reduced cross-entropy from 0.2306832 to 0.2272195; another equal step tests the continuing underconfidence trend without changing argmax predictions.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.09 * aggregate_logits
>>>>>>> REPLACE