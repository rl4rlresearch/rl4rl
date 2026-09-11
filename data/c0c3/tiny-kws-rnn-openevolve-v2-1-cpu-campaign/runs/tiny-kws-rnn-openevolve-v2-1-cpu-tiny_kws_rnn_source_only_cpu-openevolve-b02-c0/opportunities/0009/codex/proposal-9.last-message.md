MECHANISM: Single leading-frame omission

HYPOTHESIS: Processing 31 frames with the verified 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125%.

INTENDED_EDIT: Omit only the earliest input frame while retaining the remaining 31 causal frames, including the final frame.

EVIDENCE: The 99-unit 32-step model achieved 85.40% accuracy; prior 24- and 16-step schedules were too aggressive, so removing one likely low-information boundary frame is the smallest temporal-cost probe.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE