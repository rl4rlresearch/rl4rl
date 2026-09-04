MECHANISM: Center-weighted test-time ensemble refinement

HYPOTHESIS: Raising the central-view weight from 3.25 to 3.5 will improve the net validation-correct count above 9,256 by emphasizing the unshifted view, which dominates training exposure.

INTENDED_EDIT: Increase the center-to-shift ensemble ratio to 3.5:1 and update its normalization constant, preserving the best verified temperature.

EVIDENCE: Temperature scaling from 1.03 through 1.35605 never changed the 9,256 predictions and has reached its cross-entropy optimum; changing relative ensemble weights is the smallest available edit capable of improving the primary correct-count objective.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
            log_weight = math.log(3.5 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(15.0)
>>>>>>> REPLACE