MECHANISM: Uniform 24-step causal frame subsampling

HYPOTHESIS: Using 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by exactly 25%.

INTENDED_EDIT: Replace the full 32-frame schedule with a 24-frame schedule spanning the complete recording, including both endpoints.

EVIDENCE: The 128-unit GRU reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin above the target; reducing only temporal redundancy preserves model capacity while targeting the dominant 1,482,178,560 recurrent MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]
>>>>>>> REPLACE