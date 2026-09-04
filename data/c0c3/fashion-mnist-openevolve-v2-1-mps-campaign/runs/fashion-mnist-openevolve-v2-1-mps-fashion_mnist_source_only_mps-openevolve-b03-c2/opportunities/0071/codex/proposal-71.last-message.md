MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.12 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2243048.

INTENDED_EDIT: Restore the best center-view weighting and increase the post-ensemble logit multiplier from 1.09 to 1.12.

EVIDENCE: Multipliers of 1.03, 1.06, and 1.09 each preserved 9,256 correct predictions while progressively reducing cross-entropy to 0.2306832, 0.2272195, and 0.2243048; the continuing improvement motivates another equal-sized sharpening step.

<<<<<<< SEARCH
            log_weight = math.log(3.3 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.6)
=======
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.12 * aggregate_logits
>>>>>>> REPLACE