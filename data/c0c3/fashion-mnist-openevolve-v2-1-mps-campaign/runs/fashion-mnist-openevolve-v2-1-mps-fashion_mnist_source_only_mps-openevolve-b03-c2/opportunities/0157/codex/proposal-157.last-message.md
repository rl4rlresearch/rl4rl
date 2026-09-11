MECHANISM: Single-ULP downward ensemble-weight refinement

HYPOTHESIS: The next float32 center-view weight below 3.3960158824920654 will preserve 9,257 correct predictions and reduce validation cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Set the center-view weight to 3.3960156440734863 and update its exact ensemble normalization.

EVIDENCE: The immediate lower neighbor of the current weight tied its best 9,257-correct cross-entropy, while the immediate higher neighbor worsened it; continuing one float32 step downward is the finest unresolved local probe.

<<<<<<< SEARCH
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
=======
            log_weight = math.log(3.3960156440734863 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792031288146973)
>>>>>>> REPLACE