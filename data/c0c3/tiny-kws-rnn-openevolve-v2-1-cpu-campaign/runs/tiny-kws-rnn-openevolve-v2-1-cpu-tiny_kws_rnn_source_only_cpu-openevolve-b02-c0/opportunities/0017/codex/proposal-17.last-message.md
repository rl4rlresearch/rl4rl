MECHANISM: Third leading-frame omission

HYPOTHESIS: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy on frames 3–31 while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 28.3 million.

INTENDED_EDIT: Omit the first three input frames while preserving the 18-feature mel compression, 99-unit recurrent state, temporal summary, classifier, and training procedure.

EVIDENCE: Omitting frames 0–1 retained 85.03% accuracy with 30 steps, and the failed 17-feature model indicates that another input-width reduction is less promising; removing one additional leading boundary frame is the smallest remaining temporal-cost probe.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
>>>>>>> REPLACE