MECHANISM: Uniform endpoint-preserving half-rate recurrence

HYPOTHESIS: Processing 16 uniformly distributed frames instead of all 32 will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.

INTENDED_EDIT: Replace the full-frame schedule with 16 causal indices distributed across the complete recording, including both endpoints.

EVIDENCE: The current 128-unit GRU reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin for temporal subsampling while recurrent computation accounts for over 99.9% of its 1.483B inference MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(16, available_frames)
        return [
            i * (available_frames - 1) // (target_steps - 1)
            for i in range(target_steps)
        ]
>>>>>>> REPLACE