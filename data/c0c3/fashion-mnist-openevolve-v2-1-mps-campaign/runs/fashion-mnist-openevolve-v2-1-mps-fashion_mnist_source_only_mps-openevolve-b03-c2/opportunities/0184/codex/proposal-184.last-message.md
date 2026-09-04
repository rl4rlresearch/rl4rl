MECHANISM: Intermediate-dose arithmetic–geometric TTA fusion

HYPOTHESIS: A 1/16 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.2143981575012207.

INTENDED_EDIT: Replace the arithmetic-only ensemble with a 15/16 arithmetic and 1/16 center-weighted geometric probability blend, retaining the best verified compensated logit calibration.

EVIDENCE: Increasing the geometric blend from 1/64 to 1/32 preserved 9,257 correct predictions and improved cross-entropy, while 1/8 lost only one correct prediction and reduced cross-entropy further; 1/16 probes the midpoint of that ranking boundary.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.35604 * aggregate_logits
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
            arithmetic_log_probs + math.log(15.0 / 16.0),
            geometric_log_probs + math.log(1.0 / 16.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE