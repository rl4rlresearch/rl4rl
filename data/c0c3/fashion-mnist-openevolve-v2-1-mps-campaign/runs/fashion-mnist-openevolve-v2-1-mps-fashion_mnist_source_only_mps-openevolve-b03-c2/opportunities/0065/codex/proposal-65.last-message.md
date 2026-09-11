MECHANISM: Low-side center-weight refinement

HYPOTHESIS: A 3.20:1 center-to-shift inference weight will exceed 9,256 correct predictions by probing below the current peak after 3.30:1 lost one correct prediction.

INTENDED_EDIT: Reduce each center-view weight from 3.25 to 3.20 and normalize the probability mixture by 14.4.

EVIDENCE: The 3.25:1 design achieved 9,256 correct, while 3.30:1 achieved 9,255 and 3.00:1 achieved 9,253; a symmetric low-side step is the most informative local refinement.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
            log_weight = math.log(3.2 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.4)
>>>>>>> REPLACE