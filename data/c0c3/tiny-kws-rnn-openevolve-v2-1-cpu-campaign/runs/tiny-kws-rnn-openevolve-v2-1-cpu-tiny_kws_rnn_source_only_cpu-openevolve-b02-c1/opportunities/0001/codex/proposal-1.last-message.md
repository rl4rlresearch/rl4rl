MECHANISM: Uniform 2× causal frame subsampling

HYPOTHESIS: Processing every other frame will preserve at least 85% validation accuracy because the baseline’s 32-frame representation is temporally redundant, while halving recurrent steps and recurrent MACs.

INTENDED_EDIT: Change the frame schedule from all 32 frames to the 16 even-indexed frames; model capacity and training remain unchanged.

EVIDENCE: The current 128-unit GRU achieves 87.24% accuracy, providing a 2.24-point margin above the requirement, while its 32 recurrent steps account for virtually all 1.483B inference MACs.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE