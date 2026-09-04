MECHANISM: Four-frame leading-edge temporal trim

HYPOTHESIS: A 103-unit GRU processing frames 4–31 will retain at least 85% validation accuracy while reducing execution from 29 to 28 recurrent steps and recurrent MACs from 898,295,445 to approximately 867,319,740.

INTENDED_EDIT: Preserve the qualified architecture and training procedure, but omit one additional leading frame when more than four frames are available.

EVIDENCE: The 103-unit GRU remained qualified after progressively trimming one, two, and three leading frames, reaching 85.03% accuracy at 29 steps; trimming the adjacent fourth frame is the smallest untested reduction below the current cost frontier.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(3, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
>>>>>>> REPLACE