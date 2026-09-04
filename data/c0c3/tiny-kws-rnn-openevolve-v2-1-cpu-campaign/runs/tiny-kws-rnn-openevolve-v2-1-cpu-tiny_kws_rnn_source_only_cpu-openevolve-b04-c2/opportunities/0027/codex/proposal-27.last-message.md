MECHANISM: Higher-update optimization with one-step trailing trim

HYPOTHESIS: A 103-unit GRU trained at batch size 32 on frames 4–24 will retain at least 85% validation accuracy while reducing inference from 22 to 21 recurrent steps and total MACs to approximately 651.2 million.

INTENDED_EDIT: Halve the current training batch size and use a 21-frame contiguous schedule for standard 32-frame recordings, preserving the architecture and optimization procedure.

EVIDENCE: The batch-size-32 model on frames 4–25 achieved 85.89% accuracy at 22 steps, while batch size 32 recovered the corresponding batch-size-64 model from 84.79%; its 0.89-point margin supports testing the next incremental trailing-frame trim with the qualified higher-update procedure.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        return list(range(4, available_frames - 4))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
>>>>>>> REPLACE