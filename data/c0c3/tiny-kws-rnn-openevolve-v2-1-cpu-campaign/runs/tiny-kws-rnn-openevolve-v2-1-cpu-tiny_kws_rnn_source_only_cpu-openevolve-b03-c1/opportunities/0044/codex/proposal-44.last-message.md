MECHANISM: Late-boundary causal step pruning

HYPOTHESIS: Removing the final scheduled frame (index 28) will preserve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and cutting approximately 12 million dense inference MACs.

INTENDED_EDIT: Retain the informative early schedule start and proven 32/32/31 GRU capacity, but end the schedule one frame earlier.

EVIDENCE: Removing frame index 3 narrowly failed at 84.79%, while the 26-step model using index 3 achieved 85.40%; pruning the opposite boundary tests whether the retained early frame is more valuable than the final frame.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 4))
>>>>>>> REPLACE