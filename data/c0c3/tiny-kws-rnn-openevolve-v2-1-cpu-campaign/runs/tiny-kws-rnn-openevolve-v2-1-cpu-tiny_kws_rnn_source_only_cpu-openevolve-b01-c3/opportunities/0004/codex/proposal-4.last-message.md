MECHANISM: Boundary-width 16-step causal GRU

HYPOTHESIS: A 127-unit GRU using the proven 16-frame even-index schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.45% versus Reference Design 1.

INTENDED_EDIT: Select every second frame and reduce recurrent, summary, and classifier width from 128 to 127 units.

EVIDENCE: The 128-unit 16-step model reached 85.153% accuracy, while the 124-unit version narrowly missed at 84.540%; trimming only one unit tests the closest lower-cost capacity point with substantially less risk than the four-unit reduction.

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
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE