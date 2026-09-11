MECHANISM: Alternating late-tail temporal thinning

HYPOTHESIS: Omitting frames 25 and 27 while retaining frames 26 and 28 will preserve at least 85% validation accuracy with 24 recurrent steps, because spaced omissions maintain late temporal coverage better than contiguous tail truncation.

INTENDED_EDIT: Change the 32-frame schedule from frames 3–26 and 28 to frames 3–24, 26, and 28, reducing execution from 25 to 24 steps without changing model capacity or training.

EVIDENCE: The 25-step design passed only when frame 28 was retained and adjacent frame 27 was omitted; removing frame 28 narrowly failed, motivating another alternating omission in the locally redundant late region while preserving frames 26 and 28.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        return schedule
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        schedule = list(range(3, available_frames - 3))
        if len(schedule) >= 5:
            del schedule[-4]
        del schedule[-2]
        return schedule
>>>>>>> REPLACE