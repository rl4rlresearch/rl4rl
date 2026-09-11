MECHANISM: Midpoint center-view ensemble reweighting

HYPOTHESIS: Raising the center-view weight from 3.25 to 3.4375 will exceed 9,256 correct predictions by further reducing harmful shifted-view influence, while retaining enough translation averaging to preserve robustness.

INTENDED_EDIT: Increase the evaluation ensemble’s center-view weight to 3.4375, update its normalization constant, and retain the calibration used by the strongest center-weighted reference.

EVIDENCE: Increasing the center weight to 3.375 preserved 9,256 correct while lowering cross-entropy from 0.21445223 to 0.21442347; 3.4375 is a conservative midpoint toward the unmeasured 3.5 proposal.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.4375 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.356 * aggregate_logits
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.875)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE