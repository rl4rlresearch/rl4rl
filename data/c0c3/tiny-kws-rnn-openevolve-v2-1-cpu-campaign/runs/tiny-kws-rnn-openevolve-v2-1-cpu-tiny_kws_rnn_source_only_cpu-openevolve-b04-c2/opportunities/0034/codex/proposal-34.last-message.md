MECHANISM: Recurrent width-for-context exchange

HYPOTHESIS: A 91-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 543,926,110.

INTENDED_EDIT: Narrow the GRU and classifier from 103 to 91 units while expanding the standard schedule from 20 to 22 contiguous frames.

EVIDENCE: The 97-unit, 22-step design achieved 86.50% accuracy and 611,095,150 MACs; another six-unit width reduction preserves its temporal context and tests a structurally cheaper model against a 1.50-point accuracy margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE