MECHANISM: Symmetric boundary-frame pruning

HYPOTHESIS: The qualified 45/45/46 GRU can omit the final input frame in addition to the first while retaining at least 85% validation accuracy, reducing execution to 30 recurrent steps and approximately 658,881,860 total MACs.

INTENDED_EDIT: Process only frames 1–30 for 32-frame inputs, preserving the current recurrent backbone and eight-bin readout.

EVIDENCE: The current model retained 86.75% accuracy after removing the first boundary frame, leaving a 1.75-point margin; removing the opposite boundary is the smallest additional step-reduction probe and saves exactly one full recurrent update per example.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(1, available_frames - 1))
>>>>>>> REPLACE