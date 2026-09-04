MECHANISM: Five-frame leading-edge temporal trim

HYPOTHESIS: A 103-unit GRU processing frames 5–31 will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and recurrent MACs from 867,319,740 to approximately 836,344,035.

INTENDED_EDIT: Preserve the qualified architecture and training procedure, but omit the first five frames of standard 32-frame recordings.

EVIDENCE: The 103-unit GRU qualified after progressively trimming one through four leading frames, achieving 85.40% accuracy at 28 steps; trimming the adjacent fifth frame is the smallest untested reduction below that cost frontier.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(5, available_frames))
>>>>>>> REPLACE