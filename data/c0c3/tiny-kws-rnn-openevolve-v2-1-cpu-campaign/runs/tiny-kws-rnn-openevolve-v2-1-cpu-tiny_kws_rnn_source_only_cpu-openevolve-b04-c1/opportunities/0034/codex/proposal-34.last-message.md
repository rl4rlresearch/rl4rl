MECHANISM: Complementary boundary-frame temporal pruning

HYPOTHESIS: Processing 25 contiguous frames at indices 3–27 will achieve at least 85% validation accuracy while retaining the failed 25-step design’s approximately 277.7M total MACs.

INTENDED_EDIT: Shift the 25-frame window one frame earlier, dropping the latest boundary frame from the verified 26-frame schedule instead of the earliest.

EVIDENCE: The 26-frame window at indices 3–28 achieved 86.01% accuracy, while indices 4–28 narrowly missed at 84.91%; testing the complementary 3–27 window isolates which boundary frame is more informative at identical inference cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        start = (available_frames - steps + 1) // 2
        return list(range(start, start + steps))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(25, available_frames)
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
>>>>>>> REPLACE