MECHANISM: Seven-of-eight temporal sampling with endpoint preservation

HYPOTHESIS: Processing 28 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 12.5% versus the 32-step baseline.

INTENDED_EDIT: Select 28 unique, approximately uniform causal frame indices spanning the first through final input frames.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 83.93%; 28 steps is the most informative midpoint for locating the minimum schedule that clears 85%.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE