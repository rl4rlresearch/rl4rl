MECHANISM: Geometric-fusion boundary bisection

HYPOTHESIS: A 5229/65536 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870956420898.

INTENDED_EDIT: Replace arithmetic-only TTA fusion with a 60307/65536 arithmetic and 5229/65536 center-weighted geometric probability blend, retaining the verified compensated logit calibration.

EVIDENCE: The 1307/16384 blend retained 9,257 correct at 0.21436870956420898 cross-entropy, while the adjacent tested 2615/32768 blend lost one correct; 5229/65536 exactly bisects that unresolved interval.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.3560298681259155 * aggregate_logits
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
            arithmetic_log_probs + math.log(60307.0 / 65536.0),
            geometric_log_probs + math.log(5229.0 / 65536.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE