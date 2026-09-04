MECHANISM: Quadratic center-weight interpolation

HYPOTHESIS: A 3.25:1 center-to-shift inference weight will exceed 9,253 correct predictions by targeting the quadratic correctness peak implied by the verified 3:1, 3.5:1, and 3.75:1 results.

INTENDED_EDIT: Weight both center-view predictions by 3.25 while retaining unit weight for shifted views, and normalize the probability mixture by 14.5.

EVIDENCE: The 3:1 and 3.5:1 designs each achieved 9,253 correct, while 3.75:1 fell to 9,250; quadratic interpolation of these results places the estimated correctness peak at 3.25:1.

<<<<<<< SEARCH
        log_prob_views = []
        for view in views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_prob_views.extend((original, mirrored))

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
=======
        log_prob_views = []
        for view_index, view in enumerate(views):
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.5)
>>>>>>> REPLACE