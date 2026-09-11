MECHANISM: Dual-view width refinement at the qualified 29-step schedule

HYPOTHESIS: A 96-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 805,965,725 to approximately 790,849,920.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 98 to 96, and use the qualified 29-frame schedule.

EVIDENCE: The 97-unit 29-step design achieved 86.50% accuracy with 805,965,725 MACs; this 1.50-point margin supports testing the closest unresolved structural reduction, while the earlier 96-unit attempt timed out and supplied no negative accuracy evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 2))
>>>>>>> REPLACE