MECHANISM: Boundary-search temporal subsampling

HYPOTHESIS: Processing 23 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by about 4.2% versus the qualified 24-frame design.

INTENDED_EDIT: Replace full 32-frame recurrence with 23 unique increasing indices spanning both recording endpoints.

EVIDENCE: The 24-frame design achieved 85.77% accuracy, while 22 frames achieved 84.42%; testing 23 frames directly probes the qualification boundary with an estimated total cost near 1.066 billion MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(23, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE