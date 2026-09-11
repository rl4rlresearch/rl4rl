MECHANISM: Incremental causal prefix trimming

HYPOTHESIS: The 58-unit GRU processing the final 28 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.4% versus the verified 29-frame model.

INTENDED_EDIT: Omit the first four input frames while preserving the contiguous final 28-frame sequence and all other model and training settings.

EVIDENCE: The final-29-frame model achieved 85.28% accuracy after successive one-frame prefix trims through 31 and 30 frames remained viable; removing one additional earliest frame is the smallest direct test of the remaining temporal boundary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 3 if available_frames > 4 else 0
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 4 if available_frames > 5 else 0
        return list(range(start, available_frames))
>>>>>>> REPLACE