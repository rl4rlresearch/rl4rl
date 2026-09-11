MECHANISM: Boundary-frame temporal subsampling

HYPOTHESIS: The 58-unit GRU will exceed 85% accuracy using the final 31 contiguous frames, because this preserves the central frame omitted by the failed evenly spaced 31-frame schedule while retaining its lower MAC count and step count.

INTENDED_EDIT: Process frames 1–31 instead of all 32 frames, dropping only the earliest boundary frame while preserving the verified model and training procedure.

EVIDENCE: The 58-unit, 32-step model achieved 85.40%, while the evenly spaced 31-step variant narrowly missed at 84.79%; with 32 available frames that schedule omits an interior frame, motivating a same-cost 31-step schedule that discards a likely less-informative boundary frame instead.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(32, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(31, available_frames)
        return list(range(available_frames - steps, available_frames))
>>>>>>> REPLACE