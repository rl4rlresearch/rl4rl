MECHANISM: Incremental trailing-frame trim

HYPOTHESIS: A 103-unit GRU trained at batch size 64 on frames 4–27 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs to approximately 744,088,480.

INTENDED_EDIT: Remove frame 28 from the current contiguous schedule, processing 24 frames for standard 32-frame recordings.

EVIDENCE: The batch-size-64 frames 4–28 design achieved 86.26% accuracy at 25 steps; removing the adjacent trailing frame is the smallest untested reduction below the current qualified frontier.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 7:
            return list(range(available_frames))
        return list(range(4, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        return list(range(4, available_frames - 4))
>>>>>>> REPLACE