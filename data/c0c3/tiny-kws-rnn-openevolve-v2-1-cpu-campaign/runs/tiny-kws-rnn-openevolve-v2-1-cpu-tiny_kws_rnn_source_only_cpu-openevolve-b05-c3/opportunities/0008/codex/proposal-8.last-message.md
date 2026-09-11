MECHANISM: Boundary-search width reduction on qualified temporal schedule

HYPOTHESIS: A 126-unit GRU processing the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs to approximately 1.080 billion.

INTENDED_EDIT: Use 24 uniformly distributed causal frames and reduce the GRU hidden and classifier widths from 128 to 126.

EVIDENCE: The 128-unit 24-frame design achieved 85.77%, while the 124-unit variant achieved 84.29%; testing the midpoint directly probes the narrowest qualified width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE