MECHANISM: EMA-emphasized temporal ensemble

HYPOTHESIS: Weighting the decay-0.99 EMA predictions twice as strongly as the live predictions will exceed 9,290 correct by reducing final-update noise while retaining complementary live-model information.

INTENDED_EDIT: Change the live/EMA evaluation mixture from equal weighting to a normalized 1:2 weighting without adding forward passes.

EVIDENCE: Increasing EMA decay to 0.995 reduced correct predictions from 9,290 to 9,282, showing that the temporal component materially affects accuracy; emphasizing the validated 0.99 EMA is a conservative alternative to changing its horizon.

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

        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        live_log_probabilities = collect_log_probabilities()
        log_probabilities = live_log_probabilities
        total_weight = float(len(live_log_probabilities))
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
                log_probabilities = live_log_probabilities + [
                    log_probability + math.log(2.0)
                    for log_probability in ema_log_probabilities
                ]
                total_weight += 2.0 * len(ema_log_probabilities)
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(total_weight)
>>>>>>> REPLACE