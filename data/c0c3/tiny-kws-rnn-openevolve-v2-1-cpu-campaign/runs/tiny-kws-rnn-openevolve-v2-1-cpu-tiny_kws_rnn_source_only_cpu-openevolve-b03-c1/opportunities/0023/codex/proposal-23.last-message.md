MECHANISM: Three-frame leading-boundary omission

HYPOTHESIS: Processing frames 3–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the successful 30-step model.

INTENDED_EDIT: Omit one additional leading input frame, reducing recurrent execution from 30 to 29 steps while preserving model capacity, temporal-mean readout, and training procedure.

EVIDENCE: The 80-unit GRU remained above threshold after successive reductions from 32 steps (85.64%) to 31 steps (85.52%) and 30 steps (85.28%); removing the adjacent leading boundary frame is the smallest untested temporal-cost reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
>>>>>>> REPLACE