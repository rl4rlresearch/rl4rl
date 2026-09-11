MECHANISM: Second leading-frame omission

HYPOTHESIS: The verified 19-feature, 99-unit GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 28.6 million.

INTENDED_EDIT: Omit the first two input frames instead of only the first, preserving the final 30 causal frames and all model capacity.

EVIDENCE: The current 31-step design achieved 85.52% accuracy, a 0.52-point margin above the requirement; omitting one leading frame previously retained 85.03%, indicating that one additional boundary-frame omission is the most informative low-risk temporal reduction. The attempted 18-feature design supplied no accuracy evidence because training timed out.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
>>>>>>> REPLACE