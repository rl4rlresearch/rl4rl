MECHANISM: Minimal leading-frame temporal pruning

HYPOTHESIS: Omitting only the earliest frame will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125% versus the verified 92-unit, 32-step model.

INTENDED_EDIT: Retain the verified 92-unit GRU and training procedure, but process frames 1–31 instead of all 32 frames.

EVIDENCE: The 92-unit model reached 85.89% accuracy at 32 steps, whereas uniform reduction to 24 steps was too aggressive at 83.19%; removing a single likely low-information boundary frame is the smallest direct probe of temporal compression.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
>>>>>>> REPLACE