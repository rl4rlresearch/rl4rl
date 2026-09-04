MECHANISM: Tighter activation-energy screening

HYPOTHESIS: Restricting the step-22 classifier check to the highest-energy eighth will retain at least 12 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,543,139.

INTENDED_EDIT: Reduce the zero-MAC step-22 candidate screen from one quarter to one eighth of each active batch while leaving all certified checks from step 23 onward unchanged.

EVIDENCE: The highest-energy quartile retained all 14 exits produced by the unscreened step-22 check and reduced total MACs to 767,543,139; halving its classifier calls remains a strict improvement even if up to two of those exits are deferred.

<<<<<<< SEARCH
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
=======
    @staticmethod
    def _high_energy_eighth(averaged: torch.Tensor) -> torch.Tensor:
        energy = averaged.square().mean(dim=1)
        candidate_count = max(1, (averaged.shape[0] + 7) // 8)
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
            candidates = self._high_energy_eighth(averaged)
>>>>>>> REPLACE

<<<<<<< SEARCH
        early_check = completed < float(total_steps - 7)
        if bool(early_check.any()):
            averaged = summary / count.clamp_min(1.0)
            candidates = self._high_energy_quartile(averaged)
            eligible = eligible & (~early_check | candidates)
=======
        early_check = completed < float(total_steps - 7)
        if bool(early_check.any()):
            averaged = summary / count.clamp_min(1.0)
            candidates = self._high_energy_eighth(averaged)
            eligible = eligible & (~early_check | candidates)
>>>>>>> REPLACE