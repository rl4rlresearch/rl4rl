MECHANISM: Progressive trailing-edge frame omission

HYPOTHESIS: The 98-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 849,497,320 to approximately 821,223,340 and recurrent steps from 24,450 to 23,635.

INTENDED_EDIT: Retain the qualified 98-unit architecture and training procedure, but omit one additional trailing input frame for a 29-step schedule.

EVIDENCE: Omitting the final frame from the qualified 31-step 98-unit model reduced MACs while accuracy changed only from 86.50% to 86.38%; its remaining 1.38-point margin motivates testing the next trailing-edge omission.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 1))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 2))
>>>>>>> REPLACE