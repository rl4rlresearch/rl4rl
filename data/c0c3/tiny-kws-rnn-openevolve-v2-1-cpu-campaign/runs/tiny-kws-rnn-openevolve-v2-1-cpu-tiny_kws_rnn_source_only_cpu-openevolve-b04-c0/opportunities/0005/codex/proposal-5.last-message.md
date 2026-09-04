MECHANISM: Single-edge-frame causal pruning

HYPOTHESIS: Omitting only the earliest frame will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125% relative to the verified 104-unit, 32-step GRU.

INTENDED_EDIT: Process frames 1–31, preserving the latest 31 causal frames and the proven 104-unit architecture and training procedure.

EVIDENCE: Aggressive 16-frame decimation failed at 81.72%, while all 32 frames with 104 units reached 85.03%; removing just one boundary frame is the smallest temporal reduction and directly tests whether near-full resolution preserves the threshold.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE