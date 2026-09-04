MECHANISM: Fine-grained center-emphasized test-time augmentation

HYPOTHESIS: A 3.75:1 center-to-shift inference weight will exceed 9,253 correct predictions by extending the verified improvement through 3.5:1 while probing below the inconclusive 4:1 setting.

INTENDED_EDIT: Increase center-view weight from 2.0 to 3.75 and normalize the resulting probability mixture by 15.5.

EVIDENCE: Increasing center weight from 3:1 to 3.5:1 retained 9,253 correct while lowering cross-entropy from 0.2350803 to 0.2344862; the 4:1 verification timed out without accuracy evidence, motivating a 3.75:1 intermediate test.

<<<<<<< SEARCH
            log_weight = math.log(2.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(12.0)
=======
            log_weight = math.log(3.75 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(15.5)
>>>>>>> REPLACE