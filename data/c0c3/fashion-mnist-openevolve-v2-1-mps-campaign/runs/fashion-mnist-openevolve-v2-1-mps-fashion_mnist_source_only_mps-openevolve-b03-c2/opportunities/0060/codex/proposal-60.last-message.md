MECHANISM: Stronger center-emphasized test-time augmentation

HYPOTHESIS: A 4:1 center-to-shift inference weight with the verified 2.2% cosine floor will exceed 9,253 correct predictions by continuing the improvement observed from 2:1 to 3:1 center weighting.

INTENDED_EDIT: Restore the best 2.2% learning-rate floor and increase center-view inference weight to 4:1 without adding forward passes.

EVIDENCE: Increasing center weighting from 2:1 to 3:1 raised correctness from 9,249 to 9,253 and lowered cross-entropy from 0.23664 to 0.23508, motivating one further directional step.

<<<<<<< SEARCH
        log_prob_views = []
        for view in views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probs = F.log_softmax(
                self._forward_once(paired),
                dim=1,
            )
            original, mirrored = paired_log_probs.chunk(2, dim=0)
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
            log_weight = math.log(4.0 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(16.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.0225 + 0.9775 * cosine
=======
    multiplier = 0.022 + 0.978 * cosine
>>>>>>> REPLACE