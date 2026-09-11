MECHANISM: Moderate uniform causal frame subsampling

HYPOTHESIS: Using 24 uniformly spaced frames will recover at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25% versus the verified 32-step model.

INTENDED_EDIT: Replace the full 32-frame schedule with 24 unique increasing indices spanning the complete recording, including both endpoints.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while the more aggressive 16-step schedule fell to 81.72%; testing the midpoint isolates whether moderate temporal redundancy can be removed while retaining the required accuracy.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE