MECHANISM: Distributed alternating late-frame pruning

HYPOTHESIS: Processing frames 3–22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing total dense inference MACs from 282,690,900 to approximately 270,954,900.

INTENDED_EDIT: Remove frame 23 from the qualified 24-step schedule, yielding 23 recurrent steps while preserving evenly spaced late observations and the important frame-28 endpoint.

EVIDENCE: The 24-step schedule retaining frames 26 and 28 qualified at 85.15%, whereas clustering the omitted late frames missed at 84.79%; this motivates extending the successful distributed-pruning pattern by retaining frames 24, 26, and 28.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 7)) + [
            available_frames - 6,
            available_frames - 4,
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 9)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE