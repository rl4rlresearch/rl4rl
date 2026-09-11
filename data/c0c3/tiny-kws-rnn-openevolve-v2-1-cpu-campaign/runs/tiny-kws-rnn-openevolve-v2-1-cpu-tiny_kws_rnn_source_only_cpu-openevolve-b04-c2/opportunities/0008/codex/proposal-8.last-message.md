MECHANISM: Single-edge-frame temporal trim

HYPOTHESIS: Skipping only the first frame will retain at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and recurrent MACs from 991,222,560 to approximately 960,246,855.

INTENDED_EDIT: Preserve the qualified 103-unit GRU and training procedure, but omit the earliest frame from recordings containing more than two frames.

EVIDENCE: The 103-unit full-resolution model achieved 85.89% accuracy; unlike the failed 16–24-step schedules, this conservative change retains 31 of 32 frames and the complete trailing speech context.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
>>>>>>> REPLACE