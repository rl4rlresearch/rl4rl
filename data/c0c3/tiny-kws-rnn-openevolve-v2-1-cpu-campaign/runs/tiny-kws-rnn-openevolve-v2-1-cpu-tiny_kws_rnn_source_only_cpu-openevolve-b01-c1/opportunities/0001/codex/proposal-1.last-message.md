MECHANISM: Twofold causal temporal subsampling

HYPOTHESIS: Processing 16 evenly spaced frames will preserve at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Change the frame schedule from all 32 frames to every second frame, including the final frame.

EVIDENCE: The 128-unit GRU achieves 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving margin to test temporal redundancy without changing learned capacity.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames, 2))
>>>>>>> REPLACE