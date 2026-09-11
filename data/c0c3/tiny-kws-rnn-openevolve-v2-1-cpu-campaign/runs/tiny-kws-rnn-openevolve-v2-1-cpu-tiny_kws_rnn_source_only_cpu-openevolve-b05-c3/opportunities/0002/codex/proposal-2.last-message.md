MECHANISM: Uniform 20-step causal temporal subsampling

HYPOTHESIS: Increasing the failed 16-step schedule to 20 uniformly distributed frames will recover validation accuracy above 85% while reducing recurrent MACs and steps by 37.5% versus the verified 32-step model.

INTENDED_EDIT: Replace full-frame recurrence with 20 unique, increasing indices distributed across the complete recording, including both endpoints.

EVIDENCE: The 16-step model reached 84.66%, only 0.34 points below the requirement, while the 32-step model reached 87.24%; 20 steps provide additional temporal resolution with substantially lower cost than 32 steps.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            index * (available_frames - 1) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE