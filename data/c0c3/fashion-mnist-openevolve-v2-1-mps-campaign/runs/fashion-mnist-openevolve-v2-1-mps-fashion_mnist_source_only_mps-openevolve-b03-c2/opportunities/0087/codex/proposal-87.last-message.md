MECHANISM: Counter-directional center-weight ensemble refinement

HYPOTHESIS: Reducing the center-view weight from 3.25 to 3.0 will increase validation-correct above 9,256 by correcting the directional harm observed when it was raised to 3.5; the verified 1.35605 temperature will improve calibration without changing argmax predictions.

INTENDED_EDIT: Set the center-to-shift ensemble ratio to 3.0:1, update normalization from 14.5 to 14.0, and use the best verified post-ensemble multiplier of 1.35605.

EVIDENCE: Raising the center weight to 3.5 reduced validation-correct from 9,256 to 9,253, motivating exploration in the opposite direction; independently, multiplier 1.35605 achieved the best verified cross-entropy at the baseline 3.25 weighting.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.33 * aggregate_logits
=======
            log_weight = math.log(3.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.0)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE