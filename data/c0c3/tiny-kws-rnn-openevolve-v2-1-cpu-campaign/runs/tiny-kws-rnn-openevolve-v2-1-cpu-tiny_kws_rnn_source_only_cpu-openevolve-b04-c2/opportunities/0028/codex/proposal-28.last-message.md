MECHANISM: Higher-update optimization with one-step trailing trim

HYPOTHESIS: Training the 103-unit GRU at batch size 16 on frames 4–23 will retain at least 85% validation accuracy while reducing standard inference from 21 to 20 recurrent steps and total MACs to approximately 620.2 million.

INTENDED_EDIT: Halve the qualified 21-step design’s training batch size and remove its final scheduled frame, while preserving the architecture and optimization procedure.

EVIDENCE: Batch size 32 recovered the 22-step model from 84.79% to 85.89%, and the resulting 21-step model still qualified at 85.15%; another batch-size halving directly tests whether additional optimizer updates can support the next incremental trim without adding inference cost.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE