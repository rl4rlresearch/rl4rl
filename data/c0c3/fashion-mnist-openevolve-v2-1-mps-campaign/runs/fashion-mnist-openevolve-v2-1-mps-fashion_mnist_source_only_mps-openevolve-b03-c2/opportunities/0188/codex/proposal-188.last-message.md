MECHANISM: Fine-grained geometric-fusion boundary search

HYPOTHESIS: A 21/256 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.

INTENDED_EDIT: Replace arithmetic-only fusion with a 235/256 arithmetic and 21/256 center-weighted geometric probability blend, retaining the qualified compensated logit calibration.

EVIDENCE: The 5/64 blend preserved 9,257 correct predictions with 0.21436964797973632 cross-entropy, while 11/128 lost one correct prediction but further reduced cross-entropy; 21/256 is their midpoint and most directly refines the ranking boundary.

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
            arithmetic_log_probs + math.log(235.0 / 256.0),
            geometric_log_probs + math.log(21.0 / 256.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE