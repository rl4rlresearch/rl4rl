MECHANISM: Uniform causal 20-frame subsampling

HYPOTHESIS: Processing 20 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent MACs by 37.5% versus the qualified 32-step model.

INTENDED_EDIT: Replace the full 32-frame schedule with 20 unique, increasing indices spanning the first through final frame.

EVIDENCE: The 32-step model reached 87.24% accuracy, while 16 steps reached 84.29%; 20 steps is the next cost-efficient test above the observed accuracy boundary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = 20
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE