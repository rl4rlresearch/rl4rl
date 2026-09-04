MECHANISM: Uniform 29-frame causal subsampling

HYPOTHESIS: Processing 29 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.33% versus the qualified 30-step design.

INTENDED_EDIT: Replace the full 32-frame schedule with 29 unique, approximately uniform indices spanning the first and final input frames.

EVIDENCE: The 30-step design achieved 86.01% accuracy while the 28-step design achieved 83.93%; 29 steps directly tests the remaining pass/fail boundary and, if qualified, improves the dominant recurrent cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE