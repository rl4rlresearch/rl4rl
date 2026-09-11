MECHANISM: Symmetric boundary trimming

HYPOTHESIS: Processing frames 2–29 will achieve at least 85% validation accuracy while matching the failed final-28-frame model’s 310,469,360 MACs and 28 recurrent steps.

INTENDED_EDIT: Replace four-frame prefix trimming with two-frame trimming at each boundary, preserving 28 contiguous central frames.

EVIDENCE: Frames 2–31 achieved 85.52% accuracy, whereas removing two additional leading frames fell narrowly to 84.66%; removing the likely redundant final two frames instead directly tests whether retaining earlier command onset information recovers accuracy at identical cost.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 3 if available_frames > 4 else 0
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames >= 6:
            return list(range(2, available_frames - 2))
        return list(range(available_frames))
>>>>>>> REPLACE