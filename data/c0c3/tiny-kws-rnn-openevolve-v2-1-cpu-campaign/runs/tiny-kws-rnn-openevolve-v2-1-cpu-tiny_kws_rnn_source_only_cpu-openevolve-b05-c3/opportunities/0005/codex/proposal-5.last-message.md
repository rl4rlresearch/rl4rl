MECHANISM: Midpoint uniform temporal subsampling

HYPOTHESIS: Processing 22 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 8.3% versus the qualified 24-frame design.

INTENDED_EDIT: Replace full 32-frame recurrence with 22 unique increasing indices spanning the recording endpoints.

EVIDENCE: The 20-frame design reached 84.66% while the 24-frame design reached 85.77%; testing their midpoint is the most informative next search step and should cost approximately 1.020 billion total MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(22, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE