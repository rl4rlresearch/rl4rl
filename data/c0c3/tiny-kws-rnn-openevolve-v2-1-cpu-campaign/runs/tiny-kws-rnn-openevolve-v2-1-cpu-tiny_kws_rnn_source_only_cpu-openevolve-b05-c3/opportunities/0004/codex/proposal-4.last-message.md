MECHANISM: Three-of-four-frame causal recurrence

HYPOTHESIS: A 24-frame uniform schedule will recover at least 85% accuracy while reducing total inference MACs to approximately 1.112 billion, below the qualified 112-unit model.

INTENDED_EDIT: Retain the 128-unit GRU but process 24 uniformly distributed frames, including both recording endpoints.

EVIDENCE: The 20-step model missed the target by only three validation examples, while the 32-step 128-unit model reached 87.24%; 24 steps add temporal resolution while costing less than the qualified full-resolution 112-unit model.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE