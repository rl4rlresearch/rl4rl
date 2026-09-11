MECHANISM: Geometric ten-view consensus

HYPOTHESIS: Averaging ten-view logits will exceed 9,214 correct predictions by favoring classes supported consistently across the matched spatial and flip views.

INTENDED_EDIT: Replace probability-space test-time averaging with logit-space averaging, removing softmax aggregation without changing training or adding computation.

EVIDENCE: Matched cardinal-view training and flip-paired supervision produced the current 9,214-correct best; this isolates the remaining ensemble aggregation choice, while the attempted probability-ensemble training added complexity and timed out.

<<<<<<< SEARCH
        log_prob_views = []
        for view in views:
            log_prob_views.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_prob_views.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
=======
        logit_views = []
        for view in views:
            logit_views.append(self._forward_once(view))
            logit_views.append(self._forward_once(view.flip(-1)))
        return torch.stack(logit_views, dim=0).mean(dim=0)
>>>>>>> REPLACE