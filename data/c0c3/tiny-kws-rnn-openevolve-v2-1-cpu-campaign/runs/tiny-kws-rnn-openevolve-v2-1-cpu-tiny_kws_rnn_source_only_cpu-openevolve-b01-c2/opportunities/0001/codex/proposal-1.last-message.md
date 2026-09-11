MECHANISM: Two-frame causal temporal striding

HYPOTHESIS: Processing every second frame while retaining the full one-second temporal span will preserve at least 85% validation accuracy and approximately halve recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Change the frame schedule from all 32 frames to the 16 odd-indexed frames, including the final frame.

EVIDENCE: The current 128-unit GRU achieves 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving accuracy margin for a structural step reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE