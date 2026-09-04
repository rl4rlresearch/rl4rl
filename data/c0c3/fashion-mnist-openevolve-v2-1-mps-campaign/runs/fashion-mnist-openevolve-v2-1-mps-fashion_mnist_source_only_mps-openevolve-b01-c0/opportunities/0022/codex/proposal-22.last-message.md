MECHANISM: Logit-space test-time ensemble consensus

HYPOTHESIS: Averaging raw logits across live/EMA cardinal and flipped views will exceed 9,290 correct predictions by preventing a single overconfident transformed view from dominating the ensemble.

INTENDED_EDIT: Replace probability-space test-time averaging with arithmetic averaging of raw logits; training remains unchanged.

EVIDENCE: Cardinal translation-flip averaging previously improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it, showing that ensemble behavior materially affects accuracy and motivating a direct test of its aggregation rule.

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
        def collect_logits() -> list[torch.Tensor]:
            outputs = []
            for view in views:
                outputs.append(self._forward_once(view))
                outputs.append(self._forward_once(view.flip(-1)))
            return outputs

        logits = collect_logits()
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
                logits.extend(collect_logits())
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        return torch.stack(logits, dim=0).mean(dim=0)
>>>>>>> REPLACE