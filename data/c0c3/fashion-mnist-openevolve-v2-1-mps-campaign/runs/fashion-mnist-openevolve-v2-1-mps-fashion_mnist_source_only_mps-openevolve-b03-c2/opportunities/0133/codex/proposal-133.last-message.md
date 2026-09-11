MECHANISM: Conservative center-view reweighting

HYPOTHESIS: Increasing the center-view weight from 3.25 to 3.375 will exceed 9,256 correct predictions by reducing shifted-view influence on borderline examples while preserving most translation-ensemble benefit.

INTENDED_EDIT: Increase only the evaluation ensemble’s center-view weight and corresponding normalization constant; leave training and temperature calibration unchanged.

EVIDENCE: The center-biased 3.25:1 ensemble achieved 9,256 correct, while changing shift-augmentation duration or timing reduced correctness. The unverified 3.5 proposal motivates testing a smaller, more conservative step toward the center prediction.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.375 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.75)
>>>>>>> REPLACE