MECHANISM: Opposite-boundary frame trimming

HYPOTHESIS: Processing frames 3–30 will retain at least 85% validation accuracy while reducing recurrent execution from 29 to 28 steps and total inference MACs from 671,511,100 to approximately 648,378,955.

INTENDED_EDIT: Preserve the verified 60+59 GRUs and orthonormal seven-coordinate classifier, but omit the final input frame.

EVIDENCE: The current frames 3–31 design achieved 85.40% accuracy. A 28-step design that instead removed frame 3 failed, indicating that testing the opposite boundary is the cleanest way to distinguish temporal-position importance from a fundamental 29-step requirement.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))
>>>>>>> REPLACE