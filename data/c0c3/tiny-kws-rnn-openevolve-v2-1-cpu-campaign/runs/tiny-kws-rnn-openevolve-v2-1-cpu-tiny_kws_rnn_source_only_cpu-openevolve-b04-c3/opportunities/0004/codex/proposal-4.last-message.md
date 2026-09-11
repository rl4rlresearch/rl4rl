MECHANISM: Midpoint recurrent-width reduction on the qualified 28-frame schedule

HYPOTHESIS: A 126-unit GRU using 28 uniformly spaced frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.9% versus the qualified 128-unit, 28-frame design.

INTENDED_EDIT: Apply the qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 128 to 126.

EVIDENCE: At 28 frames, 128 units achieved 85.52% accuracy while 124 units achieved 84.66%; testing the midpoint width of 126 is the most informative next search within the observed qualification boundary.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
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