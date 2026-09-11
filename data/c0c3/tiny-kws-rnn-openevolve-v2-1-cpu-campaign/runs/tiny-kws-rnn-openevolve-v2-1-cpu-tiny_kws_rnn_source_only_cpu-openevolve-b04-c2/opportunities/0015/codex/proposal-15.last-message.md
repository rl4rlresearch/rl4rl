MECHANISM: Penultimate-frame temporal skip

HYPOTHESIS: Processing frames 4–29 and 31 will achieve at least 85% validation accuracy while reducing inference to 27 recurrent steps and approximately 836,344,035 recurrent MACs.

INTENDED_EDIT: Replace the current 30-step schedule with a 27-step schedule that preserves the empirically important frame 4 and final frame while omitting the locally redundant penultimate frame.

EVIDENCE: Frames 4–31 achieved 85.40% accuracy, while replacing frame 31 with frame 3 missed narrowly at 84.66%; retaining frame 31 and dropping adjacent frame 30 is the closest untested 27-step alternative.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 2)) + [available_frames - 1]
>>>>>>> REPLACE