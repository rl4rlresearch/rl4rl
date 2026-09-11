MECHANISM: Single-frame boundary trimming

HYPOTHESIS: Skipping only the first frame will preserve at least 85% validation accuracy while reducing recurrent steps from 32 to 31 and lowering exact inference MACs below the verified 99-unit full-frame model.

INTENDED_EDIT: Keep the verified 99-unit GRU and training procedure unchanged, but process frames 1–31 instead of all 32 frames.

EVIDENCE: The 99-unit 32-step model achieved 85.40% accuracy; unlike the failed 16- and 24-step schedules, this conservative change removes only one boundary frame and retains every remaining consecutive frame.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
>>>>>>> REPLACE