MECHANISM: Multi-timescale temporal weight ensembling

HYPOTHESIS: Averaging the final model with both short-horizon (0.98) and long-horizon (0.995) late-training EMAs will exceed 9,177 correct predictions by combining complementary levels of optimization-variance reduction.

INTENDED_EDIT: Track two second-half EMA states and average both of their ten-view probability predictions with the final model.

EVIDENCE: Adding one 0.98 EMA ensemble increased validation correct from 9,167 to 9,177; this directly supports extending temporal ensembling with a longer-timescale average while preserving the successful final and short-EMA predictors.

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
=======
        log_probabilities = collect_log_probabilities()
        ema_states = getattr(self, "_ema_states", ())
        if ema_states:
            live_state = dict(self.named_parameters())
            live_state.update(dict(self.named_buffers()))
            for ema_state in ema_states:
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_model = model
    optimizer._ema_state = None
=======
    optimizer._ema_model = model
    optimizer._ema_states = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        if optimizer._ema_state is None:
            optimizer._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in floating_state.items()
            }
        else:
            decay = 0.98
            with torch.no_grad():
                for name, average in optimizer._ema_state.items():
                    average.mul_(decay).add_(
                        floating_state[name].detach(), alpha=1.0 - decay
                    )
        ema_model._ema_state = optimizer._ema_state
=======
        decays = (0.98, 0.995)
        if optimizer._ema_states is None:
            optimizer._ema_states = [
                {
                    name: tensor.detach().clone()
                    for name, tensor in floating_state.items()
                }
                for _ in decays
            ]
        else:
            with torch.no_grad():
                for decay, ema_state in zip(decays, optimizer._ema_states):
                    for name, average in ema_state.items():
                        average.mul_(decay).add_(
                            floating_state[name].detach(), alpha=1.0 - decay
                        )
        ema_model._ema_states = optimizer._ema_states
>>>>>>> REPLACE