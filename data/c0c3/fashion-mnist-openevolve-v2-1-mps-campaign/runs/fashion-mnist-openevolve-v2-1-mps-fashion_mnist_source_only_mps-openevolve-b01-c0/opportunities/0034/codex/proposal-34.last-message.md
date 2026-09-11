MECHANISM: Live-emphasized temporal ensemble

HYPOTHESIS: Weighting live-model predictions twice as strongly as EMA predictions will exceed 9,290 correct predictions by shifting the ensemble opposite the EMA-heavy direction that reduced accuracy.

INTENDED_EDIT: Reuse the already-computed live log-probabilities to form a 2:1 live/EMA probability ensemble without additional forward passes.

EVIDENCE: Changing equal live/EMA weighting to 1:2 produced only 9,285 correct versus the 9,290 baseline, directly motivating a controlled test in the opposite direction.

<<<<<<< SEARCH
        log_probabilities = collect_log_probabilities()
        ema_state = getattr(self, "_ema_state", None)
=======
        live_log_probabilities = collect_log_probabilities()
        log_probabilities = list(live_log_probabilities)
        ema_state = getattr(self, "_ema_state", None)
>>>>>>> REPLACE

<<<<<<< SEARCH
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        return torch.logsumexp(
=======
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)
            log_probabilities.extend(live_log_probabilities)

        return torch.logsumexp(
>>>>>>> REPLACE