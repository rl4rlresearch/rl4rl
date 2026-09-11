MECHANISM: Uniform 2× causal frame subsampling

HYPOTHESIS: Retraining the existing 128-unit GRU on 16 uniformly spaced frames will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Replace the full 32-frame schedule with 16 increasing indices spanning the complete recording, including both endpoints.

EVIDENCE: The current model reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin above the target; recurrence accounts for 1,482,178,560 of 1,483,013,120 MACs, so temporal subsampling targets nearly all inference cost without reducing model capacity.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(16, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE