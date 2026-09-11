MECHANISM: Mild recurrent-width reduction on the qualified 28-frame schedule

HYPOTHESIS: A 124-unit GRU processing 28 uniformly spaced frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 5.7% relative to the qualified 128-unit, 28-frame design.

INTENDED_EDIT: Apply the qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 128 to 124.

EVIDENCE: The 128-unit model achieved 85.52% accuracy at 28 frames, while 24 frames fell to 83.19%; preserving the qualified temporal coverage while making a modest 3.1% width reduction targets lower MACs without crossing the observed frame-resolution boundary.

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
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE