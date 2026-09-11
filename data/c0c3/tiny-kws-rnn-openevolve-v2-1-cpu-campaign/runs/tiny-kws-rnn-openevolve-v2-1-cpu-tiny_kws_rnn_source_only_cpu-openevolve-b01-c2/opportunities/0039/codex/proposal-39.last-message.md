MECHANISM: Early-continuity-preserving temporal pruning

HYPOTHESIS: Processing frames 3–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 22 recurrent steps and approximately 259,218,900 total MACs.

INTENDED_EDIT: Remove frame 22 instead of frame 21 from the qualified 23-step schedule, preserve the proven late frames 24, 26, and 28, and remove the seven-logit classifier bias.

EVIDENCE: The 23-step schedule reached 85.52% accuracy, while the 22-step schedule that omitted frame 21 but retained frame 22 narrowly missed at 84.66%; preserving uninterrupted early recurrence through frame 21 tests whether continuity is more valuable than the final contiguous frame.

<<<<<<< SEARCH
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
=======
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE