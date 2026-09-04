MECHANISM: Width-for-time compute reallocation

HYPOTHESIS: A 128-unit GRU processing 24 uniformly spaced frames will achieve at least 85% validation accuracy while using about 4% fewer recurrent MACs than the qualified 112-unit, 32-step design.

INTENDED_EDIT: Retain the current 128-unit GRU capacity and replace the full 32-frame schedule with 24 uniformly distributed causal indices that include both endpoints.

EVIDENCE: At 112 units, reducing 32 steps to 24 lowered accuracy by 1.72 points, while the 128-unit full-sequence model exceeded the 112-unit model by 2.09 points; if these effects approximately combine, the proposed design should reach about 85.5% while reducing recurrent MACs below the qualified reference.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE