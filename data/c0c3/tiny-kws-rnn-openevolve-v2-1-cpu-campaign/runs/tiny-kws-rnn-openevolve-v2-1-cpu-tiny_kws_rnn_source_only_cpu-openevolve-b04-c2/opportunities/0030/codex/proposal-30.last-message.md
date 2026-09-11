MECHANISM: Full-span interior frame thinning

HYPOTHESIS: A 103-unit GRU trained at batch size 16 on 19 frames distributed across the qualified frames 4–24 window will retain at least 85% validation accuracy while reducing total inference MACs to approximately 589.2 million.

INTENDED_EDIT: Use batch size 16 and cap the schedule at 19 evenly distributed frames, preserving both endpoints of the qualified 21-frame temporal window.

EVIDENCE: The contiguous 20-frame batch-size-16 design qualified at 85.03%, while the 21-frame frames 4–24 design qualified at 85.15%; thinning two redundant interior frames tests a lower-cost frontier without discarding the later frames omitted by the prior trailing-trim strategy.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        window = list(range(4, available_frames - 7))
        if len(window) <= 19:
            return window
        return [
            window[(i * (len(window) - 1) + 9) // 18]
            for i in range(19)
        ]
>>>>>>> REPLACE