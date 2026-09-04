MECHANISM: Probability-space test-time augmentation ensemble

HYPOTHESIS: Averaging class probabilities across the existing ten validation views will exceed 9,166 correct predictions by limiting the influence of any single overconfident transformed view.

INTENDED_EDIT: Replace weighted logit averaging during evaluation with weighted probability averaging, returning calibrated log-probabilities as valid logits; training remains unchanged.

EVIDENCE: The zero-DC adaptive-filter model reached 9,166 correct, while subsequent filter constraints, expanded support, and loss changes did not improve it, motivating an orthogonal refinement to the existing multi-view inference path.

<<<<<<< SEARCH
        logits = []
        for view, weight in zip(views, view_weights):
            logits.append(self._forward_once(view) * weight)
            logits.append(self._forward_once(view.flip(-1)) * weight)
        return 1.034951171875 * torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
=======
        probabilities = []
        for view, weight in zip(views, view_weights):
            probabilities.append(
                F.softmax(self._forward_once(view), dim=1) * weight
            )
            probabilities.append(
                F.softmax(self._forward_once(view.flip(-1)), dim=1) * weight
            )
        mean_probabilities = torch.stack(probabilities, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
        return 1.034951171875 * mean_probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE