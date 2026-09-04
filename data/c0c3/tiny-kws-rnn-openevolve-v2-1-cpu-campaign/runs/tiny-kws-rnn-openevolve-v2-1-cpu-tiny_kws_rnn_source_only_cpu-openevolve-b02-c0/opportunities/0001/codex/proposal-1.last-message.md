MECHANISM: Uniform causal frame decimation

HYPOTHESIS: Processing the 16 odd-indexed frames will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps from the 87.24%-accurate baseline.

INTENDED_EDIT: Change the causal frame schedule from all 32 frames to every second frame, preserving the final frame.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving a 2.24-point margin for temporal decimation.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE