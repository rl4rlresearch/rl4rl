MECHANISM: Endpoint-preserving capacity-for-time exchange

HYPOTHESIS: A 61-unit GRU processing frame 3 and frames 5–28 will retain at least 85% accuracy while reducing total inference MACs below the qualified 306,309,600-MAC design to approximately 303,211,785.

INTENDED_EDIT: Widen the GRU by one unit and execute 25 steps, preserving the informative frame-28 endpoint while omitting the likely redundant early frame 4.

EVIDENCE: The 60-unit frames-3–28 design qualified at 85.40%, whereas removing frame 28 fell to 84.17%; retaining that endpoint and exchanging one early frame for modest added capacity directly targets the observed boundary, while the 58-to-60 width evidence showed a 1.35-point accuracy gain.

<<<<<<< SEARCH
        self.hidden_size = 60
=======
        self.hidden_size = 61
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 2))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [3] + list(range(5, available_frames - 3))
>>>>>>> REPLACE