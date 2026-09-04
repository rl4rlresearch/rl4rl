MECHANISM: Incremental trailing-frame trim at the 86-unit frontier

HYPOTHESIS: The 86-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 468,621,740.

INTENDED_EDIT: Remove frame 25 from the standard schedule while preserving valid schedules for shorter inputs.

EVIDENCE: The current 86-unit, 22-step model achieved 86.75% accuracy, leaving 1.75 points of margin; the analogous 97-unit trim from 22 to 21 steps retained 86.01% accuracy and reduced MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
>>>>>>> REPLACE