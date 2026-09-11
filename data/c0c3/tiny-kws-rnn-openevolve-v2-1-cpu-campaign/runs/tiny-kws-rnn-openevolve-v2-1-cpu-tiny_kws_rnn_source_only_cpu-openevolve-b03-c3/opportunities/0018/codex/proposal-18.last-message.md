MECHANISM: Single-edge-frame causal truncation

HYPOTHESIS: Skipping only the first log-mel frame will preserve at least 85% validation accuracy while reducing execution from 32 to 31 recurrent steps and total inference MACs from 776,225,560 to approximately 751,998,055.

INTENDED_EDIT: Keep the qualified 49/48/48 GRU capacity and training procedure unchanged, but begin the causal frame schedule at index 1.

EVIDENCE: The current full-frame model achieved 85.52% accuracy, providing a four-example margin; the failed 24- and 16-step schedules motivate testing the smallest possible temporal reduction rather than another aggressive subsampling scheme.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = 1 if available_frames > 2 else 0
        return list(range(start, available_frames))
>>>>>>> REPLACE