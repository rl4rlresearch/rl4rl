MECHANISM: Uniform 7:8 causal frame subsampling

HYPOTHESIS: Processing 28 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 12.5% versus the 32-step baseline.

INTENDED_EDIT: Use 28 unique, monotonically increasing frame indices spanning the complete recording, including the first and final frames.

EVIDENCE: The 32-frame model achieved 87.24% accuracy, while 24 frames achieved 84.42%; retaining half of the four-frame gap is the most informative lower-cost test with a safer accuracy margin.

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