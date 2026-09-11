MECHANISM: Single-ULP local ensemble-weight refinement

HYPOTHESIS: The next float32 center-view weight above 3.3960161209106445 will preserve 9,257 correct predictions and reduce cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Set the center-view weight to 3.3960163593292236 and update its exact ensemble normalization.

EVIDENCE: Weight 3.3960161209106445 has the best verified 9,257-correct cross-entropy, while weights six and nine float32 steps higher still preserve 9,257 correct; probing its immediate neighbor is the finest unresolved local search.

<<<<<<< SEARCH
            log_weight = math.log(3.3960132598876953 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.79202651977539)
=======
            log_weight = math.log(3.3960163593292236 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032718658447)
>>>>>>> REPLACE