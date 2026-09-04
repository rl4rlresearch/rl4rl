MECHANISM: Edge-span-preserving interior frame thinning

HYPOTHESIS: Processing 27 frames while retaining both boundary frames 4 and 31 and omitting only redundant adjacent frame 5 will preserve at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.6% versus the verified 28-step model.

INTENDED_EDIT: Keep the verified 92-unit GRU and training procedure, but change the schedule from frames 4–31 to frames 4 and 6–31.

EVIDENCE: Both 27-step boundary-pruning schedules failed, while frames 4–31 achieved 85.40%; this suggests preserving the full successful temporal span and testing redundancy within that span is more promising than removing either endpoint.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return [4] + list(range(6, available_frames))
>>>>>>> REPLACE