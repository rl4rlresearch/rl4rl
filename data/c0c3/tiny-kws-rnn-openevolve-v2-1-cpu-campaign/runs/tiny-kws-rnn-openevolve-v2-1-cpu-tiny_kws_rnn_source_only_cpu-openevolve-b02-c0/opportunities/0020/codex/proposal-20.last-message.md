MECHANISM: Penultimate-frame omission with endpoint preservation

HYPOTHESIS: Processing frames 2–29 and 31 will retain at least 85% validation accuracy while reducing execution from 30 to 29 recurrent steps and recurrent MACs by approximately 28.3 million.

INTENDED_EDIT: Omit frame 30 while preserving the earliest necessary frame and the final recurrent update.

EVIDENCE: Both 29-step boundary omissions narrowly failed at 84.66%, whereas frames 2–31 reached 85.03%; removing the penultimate frame tests temporal redundancy without losing either endpoint or the final context update.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 2)) + [available_frames - 1]
>>>>>>> REPLACE