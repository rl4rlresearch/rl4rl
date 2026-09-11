MECHANISM: Endpoint-preserving 15-step uniform causal sampling

HYPOTHESIS: Sampling 15 frames uniformly across the complete 32-frame recording will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 6.25% versus the qualified 16-step design.

INTENDED_EDIT: Replace full-frame processing with a 15-index schedule that includes both the first and final available frames.

EVIDENCE: The 16-step uniform-striding design achieved 85.153% accuracy with 741,089,280 recurrent MACs, showing substantial temporal redundancy; distributing one fewer step across the full recording may preserve coverage better than simply truncating the even-index schedule.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(15, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE