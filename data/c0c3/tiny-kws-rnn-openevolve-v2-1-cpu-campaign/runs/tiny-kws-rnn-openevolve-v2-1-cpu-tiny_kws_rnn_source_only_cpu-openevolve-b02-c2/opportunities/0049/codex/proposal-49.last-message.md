MECHANISM: Endpoint-preserving temporal thinning

HYPOTHESIS: Retaining frame 4 for early context and frame 28 for late evidence while omitting adjacent frame 5 will preserve at least 85% validation accuracy with 24 recurrent steps, reducing total inference MACs from 519,679,860 to approximately 498,936,480.

INTENDED_EDIT: Change the 25-step frames-4–28 schedule to the noncontiguous 24-step schedule containing frame 4 and frames 6–28, while preserving the qualified 84-unit, 17-band recurrent model and training procedure.

EVIDENCE: Frames 4–28 qualified at 85.03%, whereas frames 5–28 failed at 82.33% even with 85 hidden units, suggesting frame 4 supplies important early context. Earlier opposite-edge experiments also showed that retaining the latest frame recovered accuracy, motivating preservation of both endpoints while testing redundancy in their adjacent early frames.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [4, *range(6, available_frames - 3)]
>>>>>>> REPLACE