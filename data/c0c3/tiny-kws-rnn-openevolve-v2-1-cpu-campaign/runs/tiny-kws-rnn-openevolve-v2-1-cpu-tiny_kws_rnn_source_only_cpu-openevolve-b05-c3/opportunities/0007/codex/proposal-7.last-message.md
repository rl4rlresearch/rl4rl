MECHANISM: Width-reduced three-of-four-frame causal GRU

HYPOTHESIS: A 124-unit GRU processing the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 1.112 billion to approximately 1.049 billion.

INTENDED_EDIT: Apply the proven 24-frame uniform schedule and reduce the GRU hidden and classifier widths from 128 to 124.

EVIDENCE: The 128-unit 24-frame design achieved 85.77% accuracy, while reducing width all the way to 112 at full resolution still achieved 85.15%; a modest four-unit reduction should fit within the qualified 24-frame design’s 0.77-point margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
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