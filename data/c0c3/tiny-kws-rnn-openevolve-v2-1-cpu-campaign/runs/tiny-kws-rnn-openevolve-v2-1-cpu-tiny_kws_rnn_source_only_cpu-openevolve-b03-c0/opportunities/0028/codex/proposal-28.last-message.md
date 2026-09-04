MECHANISM: Earliest-frame recurrent step pruning

HYPOTHESIS: Omitting only input frame 0 will preserve at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and cutting recurrent MACs by 3.125%.

INTENDED_EDIT: Keep the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged, but schedule frames 1–31 for 32-frame inputs.

EVIDENCE: The current design passes at 85.52% accuracy; removing one recurrent step preserves all learned dimensions and the remaining 31-frame causal sequence, while offering a larger structural MAC reduction than another single-feature or classifier-coordinate reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
>>>>>>> REPLACE