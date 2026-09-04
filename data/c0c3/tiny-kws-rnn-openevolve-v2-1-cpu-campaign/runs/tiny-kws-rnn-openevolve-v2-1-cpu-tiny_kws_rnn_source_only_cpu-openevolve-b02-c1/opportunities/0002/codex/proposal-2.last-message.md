MECHANISM: Uniform 24-step causal frame subsampling

HYPOTHESIS: Using 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and MACs by 25% versus the verified 32-step baseline.

INTENDED_EDIT: Replace the full 32-frame schedule with 24 uniformly spaced indices spanning the complete recording.

EVIDENCE: The 32-step model reached 87.24% accuracy, while aggressive 16-step subsampling reached 84.17%; testing the midpoint directly brackets the accuracy-efficiency boundary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE