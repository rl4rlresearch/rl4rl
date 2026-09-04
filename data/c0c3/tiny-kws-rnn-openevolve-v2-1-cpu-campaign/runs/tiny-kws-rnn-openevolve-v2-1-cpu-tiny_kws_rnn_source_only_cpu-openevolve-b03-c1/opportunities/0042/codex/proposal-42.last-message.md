MECHANISM: Boundary-frame causal step reduction

HYPOTHESIS: Removing the earliest scheduled boundary frame will preserve at least 85% validation accuracy with the proven 32/32/31 GRU widths while reducing recurrent execution from 27 to 26 steps and total inference MACs by approximately 12.0 million.

INTENDED_EDIT: Start the contiguous frame schedule at index 3 instead of index 2, eliminating one likely low-information boundary frame without reducing recurrent capacity.

EVIDENCE: The 32/32/31 model achieved 85.40% accuracy, whereas reducing recurrent width to 32/31/31 failed; retaining the successful widths and testing a single boundary-step reduction is the most informative alternative structural reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
>>>>>>> REPLACE