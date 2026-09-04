MECHANISM: Single-frame causal prefix trimming

HYPOTHESIS: The 58-unit GRU processing the final 31 frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1%.

INTENDED_EDIT: Omit only the earliest input frame while preserving the remaining contiguous causal sequence, model width, optimizer, and training procedure.

EVIDENCE: The current 58-unit, batch-96 model achieved 85.77% accuracy over all 32 frames; prior 26–27-frame reductions were unreliable, motivating the smallest possible temporal reduction at the likely low-information recording boundary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 1 if available_frames > 2 else 0
        return list(range(start, available_frames))
>>>>>>> REPLACE