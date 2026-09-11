MECHANISM: Full-span late-frame substitution

HYPOTHESIS: Processing frames 4–22 and 24 with the qualified 86-unit GRU will preserve at least 85% accuracy while reducing inference to 20 recurrent steps and approximately 446,333,120 total MACs.

INTENDED_EDIT: Keep both endpoints of the qualified frames 4–24 window while omitting the redundant penultimate frame, replacing the unsuccessful contiguous 20-frame schedules.

EVIDENCE: Removing frame 24 missed qualification by only one validation example, while removing frame 4 performed worse; this suggests retaining the final frame and dropping an adjacent interior frame is the most targeted unresolved 20-step schedule.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        schedule = list(range(4, available_frames - 7))
        if len(schedule) > 20:
            return schedule[:19] + [schedule[-1]]
        return schedule
>>>>>>> REPLACE