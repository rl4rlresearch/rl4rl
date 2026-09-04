MECHANISM: Endpoint-preserving opposite-edge temporal thinning

HYPOTHESIS: Retaining informative endpoint frames 4 and 28 while omitting frame 27 will maintain at least 85% validation accuracy with 24 recurrent steps, reducing total inference MACs to approximately 498,936,480.

INTENDED_EDIT: Replace the 26-step frames-3–28 schedule with the 24-step schedule containing frames 4–26 and 28.

EVIDENCE: Frames 4–28 qualified at 85.03%; preserving frame 4 improved the failed 24-step attempt, while prior opposite-edge experiments showed frame 28 was more valuable than earlier boundary frames. Testing omission of frame 27 is the complementary endpoint-preserving reduction.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 5)) + [available_frames - 4]
>>>>>>> REPLACE