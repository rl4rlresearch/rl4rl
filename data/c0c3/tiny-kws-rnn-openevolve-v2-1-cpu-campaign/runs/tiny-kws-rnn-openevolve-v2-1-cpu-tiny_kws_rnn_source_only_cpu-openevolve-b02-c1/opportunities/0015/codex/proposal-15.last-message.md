MECHANISM: Single-frame boundary trimming on asymmetric block-diagonal recurrence

HYPOTHESIS: The verified 60+59 GRU will retain at least 85% validation accuracy when skipping only the first frame, while reducing recurrent steps from 32 to 31 and total inference MACs below 741,004,520.

INTENDED_EDIT: Preserve the successful asymmetric parallel-GRU model and training procedure, but process frames 1–31 instead of all 32 frames.

EVIDENCE: The current 60+59 model achieved 86.50% accuracy, providing a 1.50-point margin, and prior first-frame trimming preserved 85.03% accuracy in the 99-unit GRU while reducing its recurrent execution to 31 steps.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE