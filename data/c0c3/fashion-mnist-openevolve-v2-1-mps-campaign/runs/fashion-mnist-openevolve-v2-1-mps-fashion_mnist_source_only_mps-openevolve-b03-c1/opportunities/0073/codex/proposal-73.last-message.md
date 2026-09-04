MECHANISM: Confidence-normalized test-time augmentation ensemble

HYPOTHESIS: Weighted averaging of per-view class probabilities will exceed 9,162 correct predictions by preventing an overconfident shifted or flipped view from dominating the ten-view ensemble.

INTENDED_EDIT: Replace weighted raw-logit averaging during evaluation with weighted arithmetic probability averaging, returned as temperature-scaled log probabilities; training and parameter count remain unchanged.

EVIDENCE: Dual-statistic attention reached 9,162 correct, while subsequent attention refinements consistently regressed or timed out, motivating an orthogonal refinement of the existing multi-view decision rule without increasing training cost.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
        logits = []
        for view, weight in zip(views, view_weights):
            logits.append(self._forward_once(view) * weight)
            logits.append(self._forward_once(view.flip(-1)) * weight)
        return 1.034951171875 * torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
=======
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
        weighted_log_probabilities = []
        for view, weight in zip(views, view_weights):
            log_weight = math.log(weight)
            weighted_log_probabilities.append(
                F.log_softmax(self._forward_once(view), dim=-1) + log_weight
            )
            weighted_log_probabilities.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=-1)
                + log_weight
            )
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(weighted_log_probabilities, dim=0), dim=0
        ) - math.log(2.0 * sum(view_weights))
        return 1.034951171875 * ensemble_log_probabilities
>>>>>>> REPLACE