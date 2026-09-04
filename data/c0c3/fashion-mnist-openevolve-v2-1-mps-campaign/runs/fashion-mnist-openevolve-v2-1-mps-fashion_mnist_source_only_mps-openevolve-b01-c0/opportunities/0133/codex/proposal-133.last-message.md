MECHANISM: Accuracy-preserving canonical-view micro-upweighting

HYPOTHESIS: Raising each unshifted view’s weight to 1.0625 will retain 9,290 correct predictions while lowering validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Give the original and horizontally flipped original views 1.0625× weight, with shifted views remaining at 1×.

EVIDENCE: Weight 1.5 lowered cross-entropy to 0.20231712341308594 but lost six correct predictions; this one-eighth step toward that setting tests whether some calibration benefit is available before accuracy changes.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )

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
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        view_weights = (1.0625, 1.0, 1.0, 1.0, 1.0)
        ensemble_normalizer = math.log(2.0 * sum(view_weights))

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
        ) - ensemble_normalizer
>>>>>>> REPLACE

<<<<<<< SEARCH
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(len(ema_log_probabilities))
=======
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - ensemble_normalizer
>>>>>>> REPLACE