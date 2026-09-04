MECHANISM: Incremental trailing-frame trim at the 23-step frontier

HYPOTHESIS: The 103-unit GRU trained at batch size 64 on frames 4–25 will retain at least 85% validation accuracy while reducing standard inference from 23 to 22 recurrent steps and total MACs to approximately 682.1 million.

INTENDED_EDIT: Remove frame 26 from the qualified contiguous schedule, processing 22 frames for standard 32-frame recordings while preserving safe schedules for shorter inputs.

EVIDENCE: The batch-size-64 model on frames 4–26 achieved 85.52% accuracy at 23 steps; removing the adjacent trailing frame is the smallest untested structural reduction below the qualified cost frontier.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE