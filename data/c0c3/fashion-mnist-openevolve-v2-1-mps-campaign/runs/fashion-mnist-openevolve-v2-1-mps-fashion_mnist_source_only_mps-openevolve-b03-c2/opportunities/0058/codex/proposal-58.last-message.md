MECHANISM: Center-emphasized test-time augmentation

HYPOTHESIS: Restoring the best 2.2% cosine floor and giving the center view twice each shifted view’s inference weight will exceed 9,249 correct predictions by reducing influence from less frequently trained translations.

INTENDED_EDIT: Use the verified 2.2% terminal learning-rate floor and change ten-view probability averaging to a normalized 2:1 center-to-shift weighting without additional forward passes.

EVIDENCE: The 2.2% floor achieved the best result at 9,249 correct, while nearby floor refinements all regressed. The center crop supplies 60% of training batches but currently receives only 20% of aggregate inference weight, motivating a new inference-weighting test while retaining the strongest schedule.

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
            log_weight = math.log(2.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(12.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.0215 + 0.9785 * cosine
=======
    multiplier = 0.022 + 0.978 * cosine
>>>>>>> REPLACE