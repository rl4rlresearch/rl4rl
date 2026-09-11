MECHANISM: Three-of-four temporal sampling with endpoint preservation

HYPOTHESIS: Processing 24 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25% versus the 32-step baseline.

INTENDED_EDIT: Use 24 unique, approximately uniform causal frame indices spanning the complete recording, including the first and final frames.

EVIDENCE: The 32-step model achieved 87.24% accuracy, while aggressive 16-step decimation fell to 81.60%; testing the midpoint isolates whether moderate temporal reduction preserves the required accuracy.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 24:
            return list(range(available_frames))
        return [
            (step * (available_frames - 1)) // 23
            for step in range(24)
        ]
>>>>>>> REPLACE