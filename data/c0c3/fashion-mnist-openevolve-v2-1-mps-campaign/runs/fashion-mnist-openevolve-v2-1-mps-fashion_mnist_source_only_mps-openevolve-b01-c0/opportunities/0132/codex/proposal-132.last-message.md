MECHANISM: Midpoint canonical-view downweighting

HYPOTHESIS: Setting the original-view pair’s weight to 0.75 will correct at least one shift-sensitive error and exceed 9,290 validation-correct predictions.

INTENDED_EDIT: Reduce each unshifted TTA view’s weight from 1.0 to 0.75 while preserving shifted views, live/EMA mixing, and temperature calibration.

EVIDENCE: Weight 0.5 retained 9,289 correct while weight 1.5 fell to 9,284; this asymmetry motivates testing the unmeasured midpoint between the stronger 0.5 result and the 9,290-correct baseline.

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
        def collect_log_probabilities() -> list[torch.Tensor]:
            outputs = []
            for view_index, view in enumerate(views):
                log_weight = math.log(0.75) if view_index == 0 else 0.0
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
        ) - math.log(9.5)
>>>>>>> REPLACE

<<<<<<< SEARCH
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(len(ema_log_probabilities))
=======
            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(9.5)
>>>>>>> REPLACE