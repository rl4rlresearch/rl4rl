MECHANISM: Alternate-frame causal recurrence

HYPOTHESIS: Processing every other input frame will reduce recurrent MACs and executed steps by approximately 50% while retaining at least 85% validation accuracy because adjacent log-mel frames are highly redundant.

INTENDED_EDIT: Change the causal frame schedule from all 32 frames to the 16 even-indexed frames, leaving model capacity and training unchanged.

EVIDENCE: The current 128-unit GRU achieves 87.24% accuracy but executes all 32 recurrent steps and spends 1,482,178,560 MACs in recurrence, providing a 2.24-point accuracy margin for temporal subsampling.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE