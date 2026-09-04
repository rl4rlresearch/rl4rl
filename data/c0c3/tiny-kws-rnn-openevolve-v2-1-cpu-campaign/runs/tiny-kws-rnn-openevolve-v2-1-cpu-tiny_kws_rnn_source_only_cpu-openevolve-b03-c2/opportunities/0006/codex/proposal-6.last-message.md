MECHANISM: Joint moderate width reduction and uniform causal frame subsampling

HYPOTHESIS: The 120-unit GRU using 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.3% versus the current 32-step model.

INTENDED_EDIT: Change the frame schedule from all 32 frames to 30 approximately uniform causal frames spanning the complete recording.

EVIDENCE: The 120-unit model has 86.75% accuracy at 32 steps, while the 128-unit model retains 86.01% at 30 steps; these independent margins make their combination the most informative lower-cost candidate.

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