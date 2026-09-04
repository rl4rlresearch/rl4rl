MECHANISM: Downward logit-temperature refinement

HYPOTHESIS: Lowering the positive logit scale to 1.35602 will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21441890335083008.

INTENDED_EDIT: Restore the ensemble configuration from the best verified design and decrease only its accuracy-invariant final logit scale from 1.35603 to 1.35602.

EVIDENCE: Scale 1.35603 improved cross-entropy over 1.35604 while preserving 9,257 correct predictions; the next equal-sized downward probe is the most informative unresolved calibration test.

<<<<<<< SEARCH
            log_weight = math.log(3.3960158824920654 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.79203176498413)
        return 1.35605 * aggregate_logits
=======
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.35602 * aggregate_logits
>>>>>>> REPLACE