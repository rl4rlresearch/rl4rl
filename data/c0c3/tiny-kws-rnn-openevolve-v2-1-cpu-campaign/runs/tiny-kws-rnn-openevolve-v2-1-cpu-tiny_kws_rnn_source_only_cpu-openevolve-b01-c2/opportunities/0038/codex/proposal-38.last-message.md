MECHANISM: Extended alternating late-frame pruning

HYPOTHESIS: Processing frames 3–20, 22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing recurrent execution from 23 to 22 steps and total dense inference MACs from 270,954,900 to approximately 259,218,900.

INTENDED_EDIT: Replace the current 25-step schedule with a 22-step schedule that preserves the continuous early context and samples every second late frame through the proven frame-28 endpoint.

EVIDENCE: The 23-step frames 3–22, 24, 26, and 28 schedule achieved 85.52% accuracy; the prior successful transition from 24 to 23 steps showed that extending distributed late-frame pruning can remove a step without losing qualification.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 11)) + [
            available_frames - 10,
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE