MECHANISM: Robust arithmetic–geometric TTA fusion

HYPOTHESIS: Blending 1/8 weighted geometric consensus into the existing probability-average ensemble will suppress transformation-specific outlier predictions and increase validation_correct above 9,257.

INTENDED_EDIT: Keep the trained model and proven center/cardinal views unchanged, but blend their current arithmetic probability fusion with a center-weighted geometric fusion before logit calibration.

EVIDENCE: Temperature refinements repeatedly plateaued at 9,257 correct, while adding diagonal views reduced accuracy to 9,247; this motivates changing how the established views are fused without introducing weaker views.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.3560298681259155 * aggregate_logits
=======
        stacked = torch.stack(log_prob_views, dim=0)
        center_weight = 3.3960161209106445
        normalizer = 14.792032241821289
        arithmetic_logits = torch.logsumexp(stacked, dim=0) - math.log(normalizer)
        geometric_logits = (
            center_weight
            * (
                stacked[0]
                + stacked[1]
                - 2.0 * math.log(center_weight)
            )
            + stacked[2:].sum(dim=0)
        ) / normalizer
        aggregate_logits = (
            0.875 * arithmetic_logits
            + 0.125 * geometric_logits
        )
        return 1.3560298681259155 * aggregate_logits
>>>>>>> REPLACE