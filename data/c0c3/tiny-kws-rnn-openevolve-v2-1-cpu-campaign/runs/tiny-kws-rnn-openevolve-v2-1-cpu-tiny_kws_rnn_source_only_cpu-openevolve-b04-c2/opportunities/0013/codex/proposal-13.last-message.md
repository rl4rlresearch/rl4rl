MECHANISM: Asymmetric edge trimming that preserves the apparent speech-onset boundary

HYPOTHESIS: A 103-unit GRU processing frames 4–30 will achieve at least 85% validation accuracy while reducing execution to 27 recurrent steps and recurrent MACs to approximately 836,344,035.

INTENDED_EDIT: Preserve the qualified architecture and training procedure, retain frame 4, and replace the final frame with an additional trailing-edge trim.

EVIDENCE: Frames 4–31 qualified at 85.40%, whereas frames 5–31 fell sharply to 83.68%; this suggests frame 4 is important and makes dropping the opposite edge the most informative 27-step alternative.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
>>>>>>> REPLACE