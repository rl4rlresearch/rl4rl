MECHANISM: Opposite-boundary frame omission

HYPOTHESIS: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy on frames 2–30 while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 28.3 million.

INTENDED_EDIT: Omit the final input frame while preserving frames 2–30, the 18-feature mel compression, 99-unit recurrent state, temporal summary, classifier, and training procedure.

EVIDENCE: Omitting a third leading frame narrowly failed at 84.66%, while the current frames 2–31 schedule achieved 85.03%; testing the opposite temporal boundary is the smallest distinct 29-step probe and preserves the earliest frame known to be necessary.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 1))
>>>>>>> REPLACE