MECHANISM: Late-bridge frame pruning

HYPOTHESIS: Processing frames 3–21, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.

INTENDED_EDIT: Remove frame 24 from the qualified 22-step schedule, preserving uninterrupted early recurrence through frame 21 and the proven late frames 26 and 28.

EVIDENCE: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%, while removing frame 3 failed and prior evidence identifies frames 26 and 28 as important; frame 24 is therefore the most informative remaining single-frame pruning candidate.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE