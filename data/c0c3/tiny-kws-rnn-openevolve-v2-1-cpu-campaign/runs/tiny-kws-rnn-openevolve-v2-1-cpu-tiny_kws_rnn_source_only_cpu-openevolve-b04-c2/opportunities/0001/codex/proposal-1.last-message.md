MECHANISM: Uniform causal half-rate frame scheduling

HYPOTHESIS: Processing 16 evenly distributed frames instead of all 32 will retain at least 85% validation accuracy while halving recurrent MACs from 1,482,178,560 to approximately 741,089,280 and halving executed recurrent steps.

INTENDED_EDIT: Replace the full-frame schedule with a 16-step schedule spanning the complete recording, including its first and final frames.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy, providing 2.24 percentage points of margin, while recurrent computation accounts for more than 99.9% of its 1,483,013,120 inference MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(16, available_frames)
        return [
            index * (available_frames - 1) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE