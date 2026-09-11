MECHANISM: Uniform 30-frame causal subsampling

HYPOTHESIS: Processing 30 evenly distributed frames will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 6.25% versus the qualified 32-step design.

INTENDED_EDIT: Replace the full-frame schedule with 30 unique, approximately uniform indices spanning the first and final input frames.

EVIDENCE: The 32-step model reached 87.24% accuracy, while 28 steps fell to 83.93%; 30 steps is the midpoint of the bounded pass/fail interval and the most informative next test.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE