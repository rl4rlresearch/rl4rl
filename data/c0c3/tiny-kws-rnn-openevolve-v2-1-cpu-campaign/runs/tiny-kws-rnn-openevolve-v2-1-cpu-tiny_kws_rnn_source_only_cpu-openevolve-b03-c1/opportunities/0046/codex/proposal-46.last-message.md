MECHANISM: Penultimate-frame pruning with terminal coverage preservation

HYPOTHESIS: A 24-step schedule that removes frame 26 but retains terminal frame 27 will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 12 million versus the verified 25-step model.

INTENDED_EDIT: Replace the contiguous frames 3–27 schedule with frames 3–25 plus frame 27, omitting only the penultimate scheduled frame.

EVIDENCE: The verified 25-step frames 3–27 model achieved 85.28%, while pruning terminal frame 27 fell to 84.79%; swapping the omitted frame from 27 to its highly correlated predecessor 26 directly tests whether terminal coverage, rather than a strict 25-step requirement, caused that failure.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 4))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(3, available_frames - 4))
        schedule.remove(available_frames - 6)
        return schedule
>>>>>>> REPLACE