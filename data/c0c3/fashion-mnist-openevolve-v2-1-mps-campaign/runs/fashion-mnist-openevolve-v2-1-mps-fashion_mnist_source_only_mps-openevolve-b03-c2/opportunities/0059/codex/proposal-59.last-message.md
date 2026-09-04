MECHANISM: Stronger center-emphasized test-time augmentation

HYPOTHESIS: A 3:1 center-to-shift inference weight with the verified 2.2% cosine floor will exceed 9,249 correct predictions by further reducing the influence of translations that appear less often during training.

INTENDED_EDIT: Restore the best 2.2% learning-rate floor and increase the center view’s inference weight from the verified 2:1 setting to 3:1 without adding forward passes.

EVIDENCE: The 2:1 center-weighted design tied the best 9,249 correct result while lowering cross-entropy from 0.2390 to 0.2366 versus uniform averaging, establishing center emphasis as beneficial and motivating a directional weight increase.

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
            log_weight = math.log(3.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.018 + 0.982 * cosine
=======
    multiplier = 0.022 + 0.978 * cosine
>>>>>>> REPLACE