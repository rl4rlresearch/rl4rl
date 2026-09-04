MECHANISM: Low-dose arithmetic–geometric TTA fusion

HYPOTHESIS: Blending 1/64 geometric consensus into the established arithmetic ensemble will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.2144188995361328.

INTENDED_EDIT: Retain the proven views, weights, and compensated calibration, but add a conservative geometric-consensus component to the fused probabilities.

EVIDENCE: A 1/8 geometric blend reduced cross-entropy to 0.21433232460021973 but lost one correct prediction; a much smaller blend tests whether its calibration benefit can be retained before that ranking boundary is crossed.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
=======
        stacked = torch.stack(log_prob_views, dim=0)
        arithmetic_log_probs = (
            torch.logsumexp(stacked, dim=0)
            - math.log(14.792032241821289)
        )
        center_weight = 3.3960161209106445
        geometric_logits = (
            center_weight
            * (
                stacked[0]
                + stacked[1]
                - 2.0 * math.log(center_weight)
            )
            + stacked[2:].sum(dim=0)
        ) / 14.792032241821289
        geometric_log_probs = F.log_softmax(geometric_logits, dim=1)
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(63.0 / 64.0),
            geometric_log_probs + math.log(1.0 / 64.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE