MECHANISM: Uniform causal frame decimation

HYPOTHESIS: Sampling 16 evenly spaced frames across the full recording will retain at least 85% validation accuracy while halving recurrent steps and recurrent MACs from the verified 32-step baseline.

INTENDED_EDIT: Change the frame schedule from all 32 frames to 16 uniformly distributed frames, including the final frame.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy but executes 32 steps for every example, with recurrent computation accounting for 1,482,178,560 of 1,483,013,120 total MACs; temporal decimation directly targets nearly all inference cost while preserving full-recording coverage.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(16, available_frames)
        return [
            ((index + 1) * available_frames // steps) - 1
            for index in range(steps)
        ]
>>>>>>> REPLACE