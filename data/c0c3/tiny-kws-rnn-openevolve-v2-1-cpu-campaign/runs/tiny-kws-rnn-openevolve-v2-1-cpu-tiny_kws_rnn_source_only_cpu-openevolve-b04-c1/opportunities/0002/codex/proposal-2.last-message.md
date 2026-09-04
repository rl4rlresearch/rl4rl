MECHANISM: Moderate uniform causal frame decimation

HYPOTHESIS: Sampling 26 uniformly distributed frames across the full recording will preserve at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by 18.75% versus the verified 32-step baseline.

INTENDED_EDIT: Replace the full 32-frame schedule with 26 evenly spaced, unique causal frame indices that include both endpoints.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while aggressive reduction to 16 steps achieved only 81.72%; 26 steps is a conservative interpolation that retains substantially more temporal detail while still targeting the recurrent computation responsible for virtually all inference MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE