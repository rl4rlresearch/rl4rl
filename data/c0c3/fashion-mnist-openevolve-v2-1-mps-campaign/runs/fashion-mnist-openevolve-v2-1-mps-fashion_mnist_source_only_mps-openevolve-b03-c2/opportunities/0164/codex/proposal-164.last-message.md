MECHANISM: Quadratic logit-temperature interpolation

HYPOTHESIS: A final logit scale of 1.3560271 will preserve 9,257 correct predictions and reduce validation cross-entropy below 0.21441890335083008.

INTENDED_EDIT: Restore the best verified center-view ensemble and exact normalization, then set the final logit scale near the quadratic minimum implied by the 1.35602, 1.35603, and 1.35604 results.

EVIDENCE: Scale 1.35603 produced the lowest verified cross-entropy; 1.35602 and 1.35604 were both worse, and their asymmetric degradation estimates an optimum near 1.3560271 without changing prediction ordering.

<<<<<<< SEARCH
            log_weight = math.log(3.3960163593292236 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032718658447)
        return 1.35605 * aggregate_logits
=======
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.3560271 * aggregate_logits
>>>>>>> REPLACE