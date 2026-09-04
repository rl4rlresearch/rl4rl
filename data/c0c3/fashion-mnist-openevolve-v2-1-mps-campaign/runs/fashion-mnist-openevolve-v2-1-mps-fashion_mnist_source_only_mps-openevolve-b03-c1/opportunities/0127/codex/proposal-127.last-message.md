MECHANISM: Arithmetic-probability test-time augmentation

HYPOTHESIS: Averaging class probabilities across the ten validation views will exceed 9,206 correct predictions by preventing a single unreliable transformed view from disproportionately suppressing the correct class.

INTENDED_EDIT: Replace weighted logit averaging with weighted softmax-probability averaging, returning scaled log-probabilities while preserving all views and weights.

EVIDENCE: Adding diagonal validation views previously reduced correctness to 9,155, while removing unsupported diagonal training transforms improved correctness from 9,196 to 9,206; this indicates unequal view reliability and motivates a less veto-sensitive ensemble rule.

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
        mean_probability = torch.stack(probabilities, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
        return 1.034951171875 * mean_probability.clamp_min(1e-8).log()
>>>>>>> REPLACE