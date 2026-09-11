MECHANISM: Uniform causal 24-frame subsampling

HYPOTHESIS: Processing 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by 25% versus the qualified 32-step model.

INTENDED_EDIT: Replace the full-frame schedule with 24 unique increasing indices spanning the first and final frames.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while 16- and 20-step schedules fell below 85%; 24 steps is the next cost-efficient test between the failed reduced schedules and the qualified baseline.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
>>>>>>> REPLACE