MECHANISM: Half-step center-emphasized test-time augmentation

HYPOTHESIS: A 3.5:1 center-to-shift inference weight will exceed 9,253 correct predictions by continuing the verified improvement from 2:1 to 3:1 while conservatively probing the unverified region below 4:1.

INTENDED_EDIT: Increase the center view’s inference weight from 3.0 to 3.5 and adjust the probability-mixture normalization from 14 to 15.

EVIDENCE: Moving from 2:1 to 3:1 increased validation correctness from 9,249 to 9,253 and reduced cross-entropy from 0.23664 to 0.23508; the 4:1 run timed out without a result, making 3.5:1 the most informative intermediate test.

<<<<<<< SEARCH
            log_weight = math.log(3.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.0)
=======
            log_weight = math.log(3.5 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(15.0)
>>>>>>> REPLACE