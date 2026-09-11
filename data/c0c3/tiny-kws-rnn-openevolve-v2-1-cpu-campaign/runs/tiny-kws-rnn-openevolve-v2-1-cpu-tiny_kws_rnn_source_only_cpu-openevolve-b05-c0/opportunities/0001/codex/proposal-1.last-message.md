MECHANISM: Uniform 2× causal frame striding

HYPOTHESIS: Processing 16 evenly spaced frames instead of all 32 will preserve validation accuracy at or above 85% while halving recurrent steps and recurrent MACs from the 87.24%-accurate baseline.

INTENDED_EDIT: Select the final frame of each two-frame interval, retaining full-recording coverage and the most recent frame.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps, leaving a 2.24-point margin for temporal subsampling.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE