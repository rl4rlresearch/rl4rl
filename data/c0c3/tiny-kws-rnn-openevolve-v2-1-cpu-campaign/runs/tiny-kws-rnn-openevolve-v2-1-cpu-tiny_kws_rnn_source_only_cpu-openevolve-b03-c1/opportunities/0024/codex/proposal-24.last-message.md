MECHANISM: Terminal-boundary frame omission

HYPOTHESIS: Processing frames 2–30 with the verified 80-unit GRU will achieve at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps.

INTENDED_EDIT: Preserve the model and training procedure, but omit the final frame instead of the third leading frame.

EVIDENCE: Frames 2–31 achieved 85.28%, while frames 3–31 narrowly missed at 84.91%; comparing frames 2–30 tests whether the terminal boundary carries less useful information than frame 2 at the same lower cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 1))
>>>>>>> REPLACE