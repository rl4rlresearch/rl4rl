MECHANISM: Confidence-bounded probability-space test-time augmentation

HYPOTHESIS: Averaging view probabilities instead of logits will exceed 9,166 correct predictions by preventing any overconfident shifted view from disproportionately suppressing the correct class.

INTENDED_EDIT: Keep the established center/cardinal/flip views and weights, but combine their softmax probabilities and return calibrated log-probabilities as logits.

EVIDENCE: Adjusting the center weight in either direction failed to improve the 9,166-correct ensemble, while adding diagonal views hurt; this motivates changing how the proven views are aggregated rather than changing their composition or weights.

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
                self._forward_once(view).softmax(dim=1) * weight
            )
            probabilities.append(
                self._forward_once(view.flip(-1)).softmax(dim=1) * weight
            )
        mean_probability = torch.stack(probabilities, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
        return 1.034951171875 * mean_probability.clamp_min(1e-8).log()
>>>>>>> REPLACE