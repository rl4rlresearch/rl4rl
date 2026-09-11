MECHANISM: Uniform endpoint-preserving 24-step causal recurrence

HYPOTHESIS: Processing 24 evenly distributed frames will restore validation accuracy to at least 85% while reducing recurrent MACs and executed steps by 25% versus the verified 32-step baseline.

INTENDED_EDIT: Replace the full 32-frame schedule with 24 unique increasing indices distributed across the complete recording, including both endpoints.

EVIDENCE: The 20-step result narrowly missed the threshold at 84.42%, while the 32-step baseline reached 87.24%; testing 24 steps is the most informative intermediate point for finding the lowest viable recurrent cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]
>>>>>>> REPLACE