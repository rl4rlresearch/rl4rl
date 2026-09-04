MECHANISM: Moderate uniform causal frame subsampling

HYPOTHESIS: Processing 28 uniformly distributed frames will reduce recurrent MACs and steps by 12.5% while retaining at least 85% validation accuracy.

INTENDED_EDIT: Replace the full 32-frame schedule with 28 unique increasing indices spanning the first through final frame.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while the more aggressive 24-step schedule fell to 83.93%; 28 steps tests the midpoint of this observed accuracy–compute boundary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            index * (available_frames - 1) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE