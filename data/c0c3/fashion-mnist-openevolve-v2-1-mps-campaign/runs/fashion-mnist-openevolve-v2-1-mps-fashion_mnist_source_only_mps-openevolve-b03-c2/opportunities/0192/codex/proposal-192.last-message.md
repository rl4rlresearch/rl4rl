MECHANISM: Geometric-fusion boundary bisection

HYPOTHESIS: A 327/4096 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436881332397462.

INTENDED_EDIT: Add compensated arithmetic–geometric probability fusion using the midpoint between the best qualified 163/2048 blend and the nearest accuracy-reducing 41/512 blend.

EVIDENCE: The 163/2048 blend retained 9,257 correct at 0.21436881332397462 cross-entropy, while 41/512 lost one correct; 327/4096 exactly bisects this remaining boundary.

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
            arithmetic_log_probs + math.log(3769.0 / 4096.0),
            geometric_log_probs + math.log(327.0 / 4096.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE