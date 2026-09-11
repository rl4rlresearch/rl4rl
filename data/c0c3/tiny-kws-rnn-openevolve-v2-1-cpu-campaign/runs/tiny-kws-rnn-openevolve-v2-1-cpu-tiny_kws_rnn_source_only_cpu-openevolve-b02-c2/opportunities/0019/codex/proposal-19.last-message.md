MECHANISM: Dual-view width refinement at the qualified 29-step schedule

HYPOTHESIS: A 97-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 821,223,340 to approximately 805,965,725.

INTENDED_EDIT: Retain the current 97-unit architecture and training procedure, but omit the final two input frames in addition to the earliest frame, reducing execution from 31 to 29 recurrent steps.

EVIDENCE: The 98-unit 29-step design achieved 86.99% accuracy with 821,223,340 MACs, providing substantially more margin than the current 97-unit 31-step design’s 85.03%; reducing width by one is the closest unresolved cost refinement, while prior 28-step reductions reportedly failed.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 2))
>>>>>>> REPLACE