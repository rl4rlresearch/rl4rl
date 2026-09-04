MECHANISM: Penultimate-frame pruning

HYPOTHESIS: Retaining frame 28 while omitting adjacent frame 27 will recover at least 85% accuracy at 25 recurrent steps because frame 28 distinguished the passing 26-step schedule from the narrowly failing truncation.

INTENDED_EDIT: Use frames 3–26 and 28 for standard 32-frame inputs, reducing execution from 26 to 25 steps without changing model dimensions or training.

EVIDENCE: Frames 3–28 achieved 85.40%, while removing frame 28 scored 84.91%, only one validation example below threshold; this motivates preserving frame 28 and pruning its temporally redundant predecessor instead.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        return schedule
>>>>>>> REPLACE