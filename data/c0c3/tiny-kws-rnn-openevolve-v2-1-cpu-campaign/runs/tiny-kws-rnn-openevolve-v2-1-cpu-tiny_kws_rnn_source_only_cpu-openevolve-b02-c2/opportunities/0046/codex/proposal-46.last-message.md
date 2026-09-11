MECHANISM: Adjacent leading-frame pruning

HYPOTHESIS: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 3–28, reducing execution from 27 to 26 recurrent steps and total inference MACs from 561,166,620 to approximately 540,423,240.

INTENDED_EDIT: Preserve the qualified recurrent architecture and training procedure while removing the earliest frame from the successful 27-step frames-2–28 schedule.

EVIDENCE: Frames 2–28 achieved 85.89% accuracy, whereas frames 1–27 missed at 84.91%, showing that the latest retained frame is more valuable than the earliest; pruning frame 2 is therefore the best-supported next temporal reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 2))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
>>>>>>> REPLACE