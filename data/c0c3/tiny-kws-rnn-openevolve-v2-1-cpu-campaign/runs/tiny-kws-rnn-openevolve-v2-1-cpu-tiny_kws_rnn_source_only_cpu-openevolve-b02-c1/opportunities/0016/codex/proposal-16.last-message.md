MECHANISM: Second boundary-frame trimming

HYPOTHESIS: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and total inference MACs below 717,872,375.

INTENDED_EDIT: Skip the first two input frames instead of only the first frame, preserving the model width and training procedure.

EVIDENCE: Skipping one initial frame reduced the 60+59 model by 23,132,145 recurrent MACs while retaining 86.13% accuracy, leaving a 1.13-point margin above the requirement; removing one additional boundary frame is the largest conservative cost reduction directly supported by that result.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
>>>>>>> REPLACE