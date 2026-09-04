MECHANISM: Uniform causal frame decimation

HYPOTHESIS: Processing every second frame will preserve at least 85% validation accuracy because adjacent log-mel frames are redundant, while halving recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Change the frame schedule from all 32 frames to the 16 odd-indexed frames spanning the full recording.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps, leaving a 2.24-point accuracy margin for temporal decimation.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE