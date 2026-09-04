MECHANISM: Uniform causal frame decimation

HYPOTHESIS: Processing 16 evenly spaced frames with the unchanged 128-unit GRU will retain at least 85% accuracy while nearly halving total inference MACs and reducing recurrent steps from 32 to 16.

INTENDED_EDIT: Select the second frame of each adjacent frame pair, preserving uniform coverage and the final input frame.

EVIDENCE: The current model reaches 87.24% accuracy at 32 steps and 1.483B MACs, providing 2.24 percentage points of margin for a controlled 2× temporal reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        indices = list(range(1, available_frames, 2))
        return indices if len(indices) >= 2 else list(range(available_frames))
>>>>>>> REPLACE