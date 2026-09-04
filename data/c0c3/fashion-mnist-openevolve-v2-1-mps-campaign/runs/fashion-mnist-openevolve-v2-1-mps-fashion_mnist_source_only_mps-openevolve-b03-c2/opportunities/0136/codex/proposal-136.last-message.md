MECHANISM: Binary search of the center-view ensemble weight

HYPOTHESIS: A center-view weight of 3.390625 will preserve 9,256 correct predictions while lowering validation cross-entropy below 0.2144234748840332.

INTENDED_EDIT: Move halfway between the accuracy-preserving 3.375 weight and the 3.40625 weight that lost one prediction, update normalization, and adopt the verified 1.35605 calibration.

EVIDENCE: Weight 3.375 achieved 9,256 correct at 0.21442347 cross-entropy, while 3.40625 achieved 9,255 correct at the lower 0.21441672; their midpoint is the most informative probe of the correctness boundary.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.30 * aggregate_logits
=======
            log_weight = math.log(3.390625 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.78125)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE