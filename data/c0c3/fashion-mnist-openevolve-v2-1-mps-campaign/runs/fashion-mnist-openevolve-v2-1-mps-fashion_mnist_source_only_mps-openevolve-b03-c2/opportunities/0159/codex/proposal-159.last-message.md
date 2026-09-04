MECHANISM: Accuracy-invariant logit-temperature refinement

HYPOTHESIS: Reusing the best verified center-view weight and increasing the positive logit scale to 1.35606 will preserve 9,257 correct predictions while reducing cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Restore the best center-view weight and exact normalization, then make a small upward calibration probe of the final logit scale.

EVIDENCE: Weight 3.3960161209106445 achieved the best verified 9,257-correct cross-entropy, while adjacent center-weight probes tied or worsened it; positive logit scaling preserves class ordering and isolates the remaining calibration axis.

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
        return 1.35606 * aggregate_logits
>>>>>>> REPLACE