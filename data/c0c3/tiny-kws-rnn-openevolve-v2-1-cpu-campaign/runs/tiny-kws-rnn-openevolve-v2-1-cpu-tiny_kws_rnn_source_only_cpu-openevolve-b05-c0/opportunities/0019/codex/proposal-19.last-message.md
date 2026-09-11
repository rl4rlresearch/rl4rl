MECHANISM: Incremental causal prefix trimming

HYPOTHESIS: The 58-unit GRU processing the final 30 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.2% versus the verified 31-frame model.

INTENDED_EDIT: Omit the first two input frames while preserving the contiguous final 30-frame sequence and all training settings.

EVIDENCE: Omitting only the earliest frame retained 85.77% accuracy and reduced execution to 31 steps; one additional prefix-frame removal is the smallest and most informative temporal-cost reduction, though the 0.77-point margin makes it a boundary test.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 1 if available_frames > 2 else 0
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 2 if available_frames > 3 else 0
        return list(range(start, available_frames))
>>>>>>> REPLACE