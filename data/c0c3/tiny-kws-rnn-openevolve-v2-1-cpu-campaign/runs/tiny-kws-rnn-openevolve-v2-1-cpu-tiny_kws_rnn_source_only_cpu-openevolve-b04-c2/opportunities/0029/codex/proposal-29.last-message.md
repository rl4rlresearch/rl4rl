MECHANISM: Higher-update optimization with one-step trailing trim

HYPOTHESIS: Training the 103-unit GRU at batch size 8 on frames 4–22 will retain at least 85% validation accuracy while reducing standard inference from 20 to 19 recurrent steps and total MACs to approximately 589.2 million.

INTENDED_EDIT: Halve the training batch size and remove the final scheduled frame, while preserving the qualified architecture, optimizer, augmentation, and loss.

EVIDENCE: Successive batch-size halvings enabled the 22-step and 20-step frontiers to qualify; most recently, batch size 16 produced 85.03% accuracy at 20 steps, motivating the same optimization-for-step exchange at 19 steps.

<<<<<<< SEARCH
BATCH_SIZE = 16
=======
BATCH_SIZE = 8
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 14:
            return list(range(available_frames))
        return list(range(4, available_frames - 9))
>>>>>>> REPLACE