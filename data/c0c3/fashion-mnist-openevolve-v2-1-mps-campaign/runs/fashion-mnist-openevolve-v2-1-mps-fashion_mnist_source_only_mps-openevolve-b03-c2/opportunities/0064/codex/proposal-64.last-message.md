MECHANISM: High-side center-weight refinement

HYPOTHESIS: A 3.30:1 center-to-shift inference weight with the verified 2.2% cosine floor will retain at least 9,256 correct predictions while lowering cross-entropy below 0.2347709.

INTENDED_EDIT: Restore the best 2.2% learning-rate floor and use 3.30 weight for each center prediction versus unit weight for shifted predictions, normalized by 14.6.

EVIDENCE: The 3.25:1 design achieved the best 9,256 correct; increasing to 3.5:1 reduced cross-entropy from 0.2347709 to 0.2344862 but lost three correct predictions, motivating a small high-side step that may preserve correctness while capturing some cross-entropy improvement.

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
            log_weight = math.log(3.3 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(14.6)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.0225 + 0.9775 * cosine
=======
    multiplier = 0.022 + 0.978 * cosine
>>>>>>> REPLACE