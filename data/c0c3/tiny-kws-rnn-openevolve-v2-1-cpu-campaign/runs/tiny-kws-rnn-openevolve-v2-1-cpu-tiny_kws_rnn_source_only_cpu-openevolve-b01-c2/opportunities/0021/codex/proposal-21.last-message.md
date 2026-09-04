MECHANISM: Incremental trailing-boundary pruning

HYPOTHESIS: The 60-unit GRU processing frames 3–27 will retain at least 85% validation accuracy while reducing total inference MACs from 306,309,600 to approximately 294,573,600.

INTENDED_EDIT: Remove one additional trailing frame from the causal schedule, reducing recurrent execution from 26 to 25 steps while preserving model capacity, summaries, and training.

EVIDENCE: The qualified 60-unit models retained 86.50%, 86.26%, and 85.40% accuracy across 28, 27, and 26 steps respectively; testing 25 steps is the smallest direct probe of the trailing-context boundary and saves one complete recurrent step per example.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 4))
>>>>>>> REPLACE