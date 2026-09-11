MECHANISM: Moderate-dose arithmetic–geometric TTA fusion

HYPOTHESIS: Increasing the geometric-consensus blend from 1/64 to 1/32 will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214408358001709.

INTENDED_EDIT: Restore the qualified arithmetic–geometric ensemble and compensated calibration, using a 1/32 geometric blend to probe the interval between the successful 1/64 blend and accuracy-reducing 1/8 blend.

EVIDENCE: The 1/64 blend preserved 9,257 correct and improved cross-entropy, while 1/8 lost only one prediction and improved cross-entropy further; 1/32 is a conservative increase toward that demonstrated calibration benefit.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.356029748916626 * aggregate_logits
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
            arithmetic_log_probs + math.log(31.0 / 32.0),
            geometric_log_probs + math.log(1.0 / 32.0),
        )
        return (
            1.3560298681259155 * aggregate_logits
            + 5.4186041e-9 * aggregate_logits
        )
>>>>>>> REPLACE