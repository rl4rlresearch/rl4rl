MECHANISM: Activation-energy-screened certified early exit

HYPOTHESIS: Evaluating the full classifier only for the highest-energy quartile at step 22 will retain at least five of the 14 previously observed step-22 exits, preserving accuracy at or above 85% while reducing total inference MACs below 767,879,145.

INTENDED_EDIT: Add a zero-MAC recurrent-activation energy screen at step 22, run the learned classifier only on the selected quartile, and retain the existing full certified checks from step 23 onward.

EVIDENCE: Unscreened step-22 checks saved only 14 recurrent steps and increased total MACs because all examples incurred an extra 99-to-8 classification; screening to roughly one quarter of examples breaks even if it retains five certified exits, while the conservative certificate protects predictions.

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        if bool(torch.all(count < 23.0)):
            return averaged[:, :8]
        return self.classifier(averaged)
=======
    @staticmethod
    def _high_energy_quartile(averaged: torch.Tensor) -> torch.Tensor:
        energy = averaged.square().mean(dim=1)
        candidate_count = max(1, (averaged.shape[0] + 3) // 4)
        indices = energy.topk(candidate_count, sorted=False).indices
        candidates = torch.zeros_like(energy, dtype=torch.bool)
        candidates[indices] = True
        return candidates

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        if bool(torch.all(count < 22.0)):
            return averaged[:, :8]
        if bool(torch.all(count < 23.0)):
            candidates = self._high_energy_quartile(averaged)
            logits = averaged[:, :8].clone()
            logits[candidates] = self.classifier(averaged[candidates])
            return logits
        return self.classifier(averaged)
>>>>>>> REPLACE

<<<<<<< SEARCH
        del step
        _hidden, _summary, count = state
        completed = count[:, 0]
        eligible = (completed >= float(total_steps - 7)) & (
            completed < float(total_steps)
        )
        if not bool(eligible.any()):
            return eligible

        winner = logits.argmax(dim=1)
=======
        del step
        _hidden, summary, count = state
        completed = count[:, 0]
        eligible = (completed >= float(total_steps - 8)) & (
            completed < float(total_steps)
        )
        if not bool(eligible.any()):
            return eligible

        early_check = completed < float(total_steps - 7)
        if bool(early_check.any()):
            averaged = summary / count.clamp_min(1.0)
            candidates = self._high_energy_quartile(averaged)
            eligible = eligible & (~early_check | candidates)
            if not bool(eligible.any()):
                return eligible

        winner = logits.argmax(dim=1)
>>>>>>> REPLACE