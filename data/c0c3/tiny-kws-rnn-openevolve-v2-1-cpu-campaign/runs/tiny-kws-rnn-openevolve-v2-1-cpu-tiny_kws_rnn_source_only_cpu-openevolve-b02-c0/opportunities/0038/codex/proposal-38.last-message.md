MECHANISM: Sixteenth-batch activation-energy screening

HYPOTHESIS: Restricting the step-22 classifier check to the highest-energy sixteenth will retain at least 13 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,462,355.

INTENDED_EDIT: Halve the zero-MAC step-22 candidate pool from one eighth to one sixteenth of each active batch while leaving certified checks from step 23 onward unchanged.

EVIDENCE: Highest-energy eighth screening retained all 14 exits from the unscreened step-22 check and reduced total MACs to 767,462,355; halving classifier calls remains a strict improvement if at most one exit is deferred.

<<<<<<< SEARCH
    @staticmethod
    def _high_energy_eighth(averaged: torch.Tensor) -> torch.Tensor:
        energy = averaged.square().mean(dim=1)
        candidate_count = max(1, (averaged.shape[0] + 7) // 8)
=======
    @staticmethod
    def _high_energy_sixteenth(averaged: torch.Tensor) -> torch.Tensor:
        energy = averaged.square().mean(dim=1)
        candidate_count = max(1, (averaged.shape[0] + 15) // 16)
>>>>>>> REPLACE

<<<<<<< SEARCH
            candidates = self._high_energy_eighth(averaged)
=======
            candidates = self._high_energy_sixteenth(averaged)
>>>>>>> REPLACE

<<<<<<< SEARCH
            candidates = self._high_energy_eighth(averaged)
=======
            candidates = self._high_energy_sixteenth(averaged)
>>>>>>> REPLACE