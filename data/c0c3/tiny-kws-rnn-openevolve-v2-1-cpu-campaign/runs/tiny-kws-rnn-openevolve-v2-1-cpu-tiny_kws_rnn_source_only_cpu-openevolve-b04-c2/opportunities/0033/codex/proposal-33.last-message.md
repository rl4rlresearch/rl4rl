MECHANISM: Incremental trailing-frame trim at reduced recurrent width

HYPOTHESIS: The 97-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing standard inference from 22 to 21 recurrent steps and total MACs to approximately 583.3 million.

INTENDED_EDIT: Remove frame 25 from the current schedule, processing 21 contiguous frames for standard recordings while preserving valid schedules for shorter inputs.

EVIDENCE: The current 97-unit, 22-step model achieved 86.50% accuracy, a 1.50-point margin; the analogous 103-unit batch-size-32 trim from 22 to 21 steps lost only 0.74 points and still qualified.

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