MECHANISM: Early-boundary causal step pruning

HYPOTHESIS: Removing frame index 3 will preserve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and cutting approximately 12 million dense inference MACs.

INTENDED_EDIT: Advance the contiguous frame schedule start from index 3 to index 4 while retaining the proven 32/32/31 GRU capacity and all training settings.

EVIDENCE: Removing the previous earliest boundary frame, index 2, retained 85.40% accuracy and reduced execution from 27 to 26 steps; pruning the adjacent early-boundary frame is the smallest continuation of that successful structural reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE