MECHANISM: Moderate uniform causal frame thinning

HYPOTHESIS: Using 28 uniformly spaced frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 12.5% versus the qualified 32-frame design.

INTENDED_EDIT: Replace the full 32-frame schedule with 28 unique, uniformly distributed indices spanning the complete recording.

EVIDENCE: The 32-frame model achieved 87.24% accuracy, while aggressive thinning to 24 frames achieved 83.19%; testing the midpoint of 28 frames is the most informative next cost-accuracy tradeoff.

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