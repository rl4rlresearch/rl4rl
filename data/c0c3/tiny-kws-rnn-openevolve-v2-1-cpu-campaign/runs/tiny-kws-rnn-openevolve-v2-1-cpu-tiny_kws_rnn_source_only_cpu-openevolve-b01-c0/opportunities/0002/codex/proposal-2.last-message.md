MECHANISM: Uniform 28-step causal frame subsampling

HYPOTHESIS: Processing 28 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed recurrent steps by 12.5% versus the verified 32-step model.

INTENDED_EDIT: Replace the full-frame schedule with 28 unique, increasing indices spanning the complete recording, including both endpoints.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 84.42%; testing the midpoint of this bracket is the most informative next reduction with greater accuracy margin than the failed 24-step design.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
>>>>>>> REPLACE