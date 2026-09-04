MECHANISM: Slight EMA-favoring probability ensemble

HYPOTHESIS: A normalized 49:51 live-to-EMA probability mixture will either exceed 9,290 correct predictions or preserve them while reducing validation cross-entropy below 0.20249243125915528.

INTENDED_EDIT: Compute live and EMA test-time augmentation ensembles separately, then combine their probabilities with 49% live and 51% EMA weight while retaining the best verified temperature.

EVIDENCE: Local temperature calibration is exhausted: both adjacent float32 temperatures were worse than the current 0.717143714427948 value. The only 49:51 EMA-favoring trial timed out, so its accuracy and cross-entropy effect remain unresolved and merit a direct retry.

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
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(live_log_probabilities, dim=0), dim=0
        ) - math.log(len(live_log_probabilities))

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
                ema_ensemble_log_probabilities = torch.logsumexp(
                    torch.stack(ema_log_probabilities, dim=0), dim=0
                ) - math.log(len(ema_log_probabilities))
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

            ensemble_log_probabilities = torch.logaddexp(
                ensemble_log_probabilities + math.log(0.49),
                ema_ensemble_log_probabilities + math.log(0.51),
            )

        return ensemble_log_probabilities / 0.717143714427948
>>>>>>> REPLACE