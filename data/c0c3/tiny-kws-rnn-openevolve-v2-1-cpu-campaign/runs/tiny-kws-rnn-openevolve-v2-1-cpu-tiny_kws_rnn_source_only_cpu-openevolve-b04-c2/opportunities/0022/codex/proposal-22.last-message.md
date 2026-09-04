MECHANISM: Higher-update optimization with incremental trailing-frame trim

HYPOTHESIS: A 103-unit GRU trained at batch size 64 on frames 4–28 will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs to approximately 775,064,185.

INTENDED_EDIT: Halve the training batch size and schedule 25 contiguous frames for standard 32-frame recordings by omitting frames 0–3 and 29–31.

EVIDENCE: The batch-size-64, frames 4–29 design achieved 87.36% accuracy at 26 steps, leaving 2.36 percentage points of margin; removing the adjacent trailing frame is the smallest untested reduction below that qualified frontier.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 7:
            return list(range(available_frames))
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE