MECHANISM: Stronger center-view weighting in the inference ensemble

HYPOTHESIS: Increasing the center-view weight from 3.25 to 3.5 will exceed 9,256 correct predictions by reducing the influence of shifted views on borderline examples while retaining their robustness benefits.

INTENDED_EDIT: Increase only the evaluation ensemble’s center-view weight and update its normalization constant; preserve training and temperature calibration.

EVIDENCE: Both shortening and redistributing shift augmentation reduced validation_correct, while the center-biased 3.25:1 ensemble retained 9,256 correct; a small increase isolates whether remaining errors come from over-weighting shifted predictions without perturbing the verified training trajectory.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
            log_weight = math.log(3.5 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(15.0)
>>>>>>> REPLACE