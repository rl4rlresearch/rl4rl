MECHANISM: Quadratically confidence-tilted probability pooling

HYPOTHESIS: A mild convex confidence tilt will exceed 9,210 correct predictions by emphasizing confident agreement across views without restoring the low-probability veto behavior of geometric pooling.

INTENDED_EDIT: Replace arithmetic view-probability averaging with a normalized average of `p + 0.25p²`, preserving all validation views and weights.

EVIDENCE: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling. The fractional power-mean test timed out, so this computationally cheap multiply-add tests a smaller move in the same confidence-emphasizing direction.

<<<<<<< SEARCH
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
=======
        probabilities = []
        for view, weight in zip(views, view_weights):
            probability = F.softmax(self._forward_once(view), dim=1)
            probabilities.append(
                probability * (1.0 + 0.25 * probability) * weight
            )
            probability = F.softmax(self._forward_once(view.flip(-1)), dim=1)
            probabilities.append(
                probability * (1.0 + 0.25 * probability) * weight
            )
        mean_probability = torch.stack(probabilities, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
        mean_probability = mean_probability / mean_probability.sum(
            dim=1, keepdim=True
        )
        return 1.034951171875 * mean_probability.clamp_min(1e-8).log()
>>>>>>> REPLACE