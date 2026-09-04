MECHANISM: Two-frame leading-boundary omission

HYPOTHESIS: Processing frames 2–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.2% versus the successful 31-step model.

INTENDED_EDIT: Omit one additional leading input frame, reducing recurrent execution from 31 to 30 steps while preserving model capacity, temporal-mean readout, and training procedure.

EVIDENCE: The 80-unit GRU achieved 85.52% after omitting frame 0, while more aggressive 28-frame schedules failed; removing only the adjacent likely boundary frame is the smallest informative temporal-cost reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
>>>>>>> REPLACE