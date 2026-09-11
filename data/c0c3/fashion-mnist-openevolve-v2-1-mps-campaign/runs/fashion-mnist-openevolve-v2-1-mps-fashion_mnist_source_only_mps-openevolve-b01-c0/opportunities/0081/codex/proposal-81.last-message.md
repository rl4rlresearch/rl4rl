MECHANISM: Conservative live-model ensemble-weight interpolation

HYPOTHESIS: A 50.1% live / 49.9% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249243125915528.

INTENDED_EDIT: Compute live and EMA TTA probability ensembles separately, combine them with 50.1% live weight and 49.9% EMA weight, and retain the best verified temperature.

EVIDENCE: The 51% live mixture lowered cross-entropy to 0.20248852615356444 but lost one correct prediction, whereas 49% live retained 9,290 correct but worsened cross-entropy; the prior 50.1% probe timed out, leaving this conservative interpolation unresolved.

<<<<<<< SEARCH
        log_probabilities = collect_log_probabilities()
        ema_state = getattr(self, "_ema_state", None)
        if ema_state is not None:
            live_state = dict(self.named_parameters())
            live_state.update(dict(self.named_buffers()))
            backups = {}
            with torch.no_grad():
                for name, average in ema_state.items():
                    tensor = live_state[name]
                    backups[name] = tensor.detach().clone()
                    tensor.copy_(average)
            try:
                log_probabilities.extend(collect_log_probabilities())
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return ensemble_log_probabilities / 0.717143714427948
=======
        live_log_probabilities = collect_log_probabilities()
        live_ensemble = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0), dim=0
        ) - math.log(len(live_log_probabilities))

        ensemble_log_probabilities = live_ensemble
        ema_state = getattr(self, "_ema_state", None)
        if ema_state is not None:
            live_state = dict(self.named_parameters())
            live_state.update(dict(self.named_buffers()))
            backups = {}
            with torch.no_grad():
                for name, average in ema_state.items():
                    tensor = live_state[name]
                    backups[name] = tensor.detach().clone()
                    tensor.copy_(average)
            try:
                ema_log_probabilities = collect_log_probabilities()
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

            ema_ensemble = torch.logsumexp(
                torch.stack(ema_log_probabilities, dim=0), dim=0
            ) - math.log(len(ema_log_probabilities))
            ensemble_log_probabilities = torch.logsumexp(
                torch.stack(
                    (
                        live_ensemble + math.log(0.501),
                        ema_ensemble + math.log(0.499),
                    ),
                    dim=0,
                ),
                dim=0,
            )

        return ensemble_log_probabilities / 0.717143714427948
>>>>>>> REPLACE