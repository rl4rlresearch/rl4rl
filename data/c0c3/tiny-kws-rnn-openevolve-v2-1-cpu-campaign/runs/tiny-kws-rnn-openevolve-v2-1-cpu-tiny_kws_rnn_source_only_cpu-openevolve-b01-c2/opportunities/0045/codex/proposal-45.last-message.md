MECHANISM: Interior-frame temporal pruning

HYPOTHESIS: Processing frames 3–11, 13–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.

INTENDED_EDIT: Remove only the central frame 12 from the qualified 22-step schedule, preserving both temporal boundaries of the dense early segment and all three proven late observations.

EVIDENCE: Removing frame 3 or frame 24 from the qualified schedule failed, indicating that the early boundary and late bridge are important; pruning one interior frame tests redundancy within the densely sampled 3–21 region while introducing only a single two-frame gap.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return (
            list(range(3, available_frames - 20))
            + list(range(available_frames - 19, available_frames - 10))
            + [
                available_frames - 8,
                available_frames - 6,
                available_frames - 4,
            ]
        )
>>>>>>> REPLACE