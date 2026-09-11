MECHANISM: Moderate uniform causal temporal subsampling

HYPOTHESIS: Processing 24 evenly distributed frames, including both endpoints, will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.

INTENDED_EDIT: Replace the full 32-frame schedule with 24 uniformly spaced, unique, increasing frame indices.

EVIDENCE: The 32-step model reached 87.24% accuracy, while aggressive 16-step subsampling reached 83.19%; testing the midpoint directly probes whether moderate temporal redundancy can be removed without crossing the 85% threshold.

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