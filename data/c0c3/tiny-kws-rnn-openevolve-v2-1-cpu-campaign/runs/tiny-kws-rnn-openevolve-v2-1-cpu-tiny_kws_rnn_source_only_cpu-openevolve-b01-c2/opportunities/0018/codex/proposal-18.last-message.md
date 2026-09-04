MECHANISM: Trailing-boundary pruning

HYPOTHESIS: The 60-unit GRU can process frames 3–30 and retain at least 85% validation accuracy while reducing total inference MACs from 341,517,600 to approximately 329,781,600.

INTENDED_EDIT: Keep 28 recurrent steps by removing the final boundary frame instead of the more informative frame 3.

EVIDENCE: Processing frames 4–31 missed the accuracy requirement by only one validation example (84.91%), while frames 3–31 achieved 85.40%; exchanging the final frame for frame 3 tests whether the leading frame carries more discriminative speech context.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))
>>>>>>> REPLACE