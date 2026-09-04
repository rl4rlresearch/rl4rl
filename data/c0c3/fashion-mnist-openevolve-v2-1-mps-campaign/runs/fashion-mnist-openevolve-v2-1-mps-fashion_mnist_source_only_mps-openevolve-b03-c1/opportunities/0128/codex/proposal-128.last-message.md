MECHANISM: Mild power-mean test-time pooling

HYPOTHESIS: Raising the probability-pooling power from 1 to 1.25 will exceed 9,210 correct predictions by emphasizing confident agreement while retaining the arithmetic ensemble’s resistance to low-probability vetoes.

INTENDED_EDIT: Replace arithmetic probability averaging with a class-wise power mean across the same ten weighted validation views.

EVIDENCE: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling, motivating a modest extrapolation beyond the arithmetic endpoint rather than returning toward the inferior geometric rule.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
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
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
        pool_power = 1.25
        powered_probabilities = []
        for view, weight in zip(views, view_weights):
            powered_probabilities.append(
                F.softmax(self._forward_once(view), dim=1).pow(pool_power)
                * weight
            )
            powered_probabilities.append(
                F.softmax(self._forward_once(view.flip(-1)), dim=1).pow(
                    pool_power
                )
                * weight
            )
        mean_powered_probability = torch.stack(
            powered_probabilities, dim=0
        ).sum(dim=0) / (2.0 * sum(view_weights))
        return (
            1.034951171875
            * mean_powered_probability.clamp_min(1e-8).log()
            / pool_power
        )
>>>>>>> REPLACE