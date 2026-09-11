MECHANISM: Higher-update optimization for trailing-frame trimming

HYPOTHESIS: Halving the training batch to 64 will provide enough additional optimizer updates for the 103-unit GRU on frames 4–30 to reach at least 85% accuracy while reducing total inference MACs to approximately 837,015,595.

INTENDED_EDIT: Use the previously tested 27-step contiguous schedule and halve the batch size, while preserving the architecture, optimizer, augmentation, and loss.

EVIDENCE: The 103-unit frames 4–30 model narrowly missed qualification at 84.66%, only 0.34 percentage points short; increasing width and changing the readout both performed worse, so increasing optimization updates without adding inference cost is the most direct untested intervention.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
>>>>>>> REPLACE