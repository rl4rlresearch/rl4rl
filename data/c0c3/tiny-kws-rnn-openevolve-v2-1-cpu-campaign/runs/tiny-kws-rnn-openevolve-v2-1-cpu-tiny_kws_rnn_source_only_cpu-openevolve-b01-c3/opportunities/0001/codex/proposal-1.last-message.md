MECHANISM: Uniform twofold causal temporal striding

HYPOTHESIS: Processing 16 uniformly distributed frames instead of all 32 will preserve at least 85% validation accuracy from the current 87.24% baseline while approximately halving recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Change the causal frame schedule to select every second input frame.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving accuracy margin for a structural reduction targeting temporal redundancy.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE