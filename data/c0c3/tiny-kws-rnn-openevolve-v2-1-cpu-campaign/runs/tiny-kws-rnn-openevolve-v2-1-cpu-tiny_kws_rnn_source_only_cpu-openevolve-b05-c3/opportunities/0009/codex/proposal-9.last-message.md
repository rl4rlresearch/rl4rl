MECHANISM: One-unit boundary width reduction on the qualified 24-frame GRU

HYPOTHESIS: A 127-unit GRU using the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs below the 1,112,468,480-MAC 128-unit design.

INTENDED_EDIT: Process 24 uniformly distributed causal frames and reduce the GRU hidden state, temporal summary, and classifier input from 128 to 127 units.

EVIDENCE: The 128-unit 24-frame model achieved 85.77%, whereas the 126-unit variant achieved 84.66%; testing the intervening 127-unit width directly resolves the remaining qualification boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
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