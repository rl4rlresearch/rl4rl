MECHANISM: Moderate uniform causal frame thinning

HYPOTHESIS: Using 24 uniformly distributed frames across the full one-second span will retain at least 85% validation accuracy while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.

INTENDED_EDIT: Skip every fourth frame while preserving the final frame, yielding 24 recurrent steps.

EVIDENCE: The 32-step model reached 87.24% accuracy, while aggressive 16-step striding fell to 83.19%; 24 steps directly tests the intermediate compute–accuracy point with the same temporal coverage.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [index for index in range(available_frames) if index % 4 != 0]
>>>>>>> REPLACE