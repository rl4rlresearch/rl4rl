MECHANISM: Conservative center-weight boundary refinement

HYPOTHESIS: A center-view weight of 3.3828125 will preserve all 9,256 correct predictions while lowering validation cross-entropy below 0.2144234748840332.

INTENDED_EDIT: Increase the evaluation ensemble’s center-view weight from 3.375 to 3.3828125 and update its normalization constant.

EVIDENCE: Weight 3.375 retained 9,256 correct, while 3.40625 lost one prediction despite lower cross-entropy; 3.3828125 is a conservative quarter-step toward that boundary after the intervening 3.390625 verification timed out without performance evidence.

<<<<<<< SEARCH
            log_weight = math.log(3.375 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.75)
=======
            log_weight = math.log(3.3828125 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.765625)
>>>>>>> REPLACE