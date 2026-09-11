MECHANISM: Opposite-direction canonical-view downweighting

HYPOTHESIS: Halving the unshifted view’s weight from 1.0 to 0.5 will correct at least one shift-sensitive error and exceed 9,290 validation-correct predictions.

INTENDED_EDIT: Reduce the original and horizontally flipped original views from 20% to 11.11% of the TTA ensemble while preserving all shifted views, live/EMA mixing, and temperature calibration.

EVIDENCE: Increasing canonical-view weight to 1.5 reduced validation-correct from 9,290 to 9,284; probing the opposite direction is the most informative remaining view-weight change.

<<<<<<< SEARCH
        def collect_log_probabilities() -> list[torch.Tensor]:
            outputs = []
            for view in views:
                outputs.append(
                    F.log_softmax(self._forward_once(view), dim=1)
                )
                outputs.append(
                    F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
                )
            return outputs
=======
        view_weights = (0.5, 1.0, 1.0, 1.0, 1.0)

        def collect_log_probabilities() -> list[torch.Tensor]:
            outputs = []
            for view, weight in zip(views, view_weights):
                log_weight = math.log(weight)
                outputs.append(
                    F.log_softmax(self._forward_once(view), dim=1) + log_weight
                )
                outputs.append(
                    F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
                    + log_weight
                )
            return outputs
>>>>>>> REPLACE

<<<<<<< SEARCH
        live_ensemble = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0), dim=0
        ) - math.log(len(live_log_probabilities))
=======
        live_ensemble = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0), dim=0
        ) - math.log(2.0 * sum(view_weights))
>>>>>>> REPLACE

<<<<<<< SEARCH
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(len(ema_log_probabilities))
=======
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(2.0 * sum(view_weights))
>>>>>>> REPLACE