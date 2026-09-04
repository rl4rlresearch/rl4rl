MECHANISM: Single boundary-frame omission

HYPOTHESIS: Processing frames 1–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by 3.125%.

INTENDED_EDIT: Omit only the first input frame while preserving the remaining 31 causal frames, recurrent width, temporal-mean readout, and training procedure.

EVIDENCE: The 80-unit full-rate model achieved 85.64% accuracy; although reducing to 28 frames failed, omitting one likely low-information boundary frame is a substantially more conservative temporal reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE