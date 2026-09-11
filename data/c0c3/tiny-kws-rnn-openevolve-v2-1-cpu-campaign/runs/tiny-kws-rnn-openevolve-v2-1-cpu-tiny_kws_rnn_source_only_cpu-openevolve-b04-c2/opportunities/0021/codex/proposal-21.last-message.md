MECHANISM: Higher-update optimization with one-step trailing trim

HYPOTHESIS: A 103-unit GRU trained at batch size 64 on frames 4–29 will retain at least 85% validation accuracy while reducing execution from the qualified 27-step design to 26 steps and total inference MACs to approximately 806 million.

INTENDED_EDIT: Use the qualified batch-size-64 training procedure and remove frame 30 from its contiguous schedule, processing 26 frames per standard recording.

EVIDENCE: The batch-size-64 model on frames 4–30 achieved 87.24% accuracy at 27 steps, providing 2.24 percentage points of margin; trimming one adjacent trailing frame is the smallest untested reduction below that qualified cost frontier.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 2))
>>>>>>> REPLACE