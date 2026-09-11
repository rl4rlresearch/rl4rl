MECHANISM: One-unit recurrent-width boundary search

HYPOTHESIS: A 127-unit GRU using the qualified 28-frame schedule will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.45% versus the qualified 128-unit design.

INTENDED_EDIT: Use 28 uniformly spaced causal frames and reduce the GRU state, temporal summary, and classifier width from 128 to 127.

EVIDENCE: At 28 frames, 128 units achieved 85.52% accuracy while 126 units achieved 84.66%; testing the intervening 127-unit width directly resolves the narrowest observed qualification boundary.

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
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE