MECHANISM: Uniform 28-frame causal subsampling

HYPOTHESIS: Using 28 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 12.5% versus the qualified 32-step design.

INTENDED_EDIT: Change the frame schedule from all 32 frames to 28 evenly spaced causal frames spanning the first through final input frames.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 83.93%; testing the midpoint of this bounded interval is the most informative next change and is expected to preserve the required accuracy with lower dominant recurrent cost.

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