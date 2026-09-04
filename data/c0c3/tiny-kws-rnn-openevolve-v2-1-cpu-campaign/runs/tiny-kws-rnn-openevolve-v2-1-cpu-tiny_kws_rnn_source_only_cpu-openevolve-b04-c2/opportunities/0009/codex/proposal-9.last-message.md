MECHANISM: Two-frame leading-edge temporal trim

HYPOTHESIS: A 103-unit GRU processing frames 2–31 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs from 960,246,855 to approximately 929,271,150.

INTENDED_EDIT: Use the qualified 103-unit recurrent width and omit the first two frames when more than two frames are available.

EVIDENCE: The 103-unit GRU skipping only the first frame achieved 85.64% accuracy at 31 steps, leaving 0.64 percentage points of margin; trimming one additional adjacent leading frame is the smallest untested temporal-cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(2, available_frames))
>>>>>>> REPLACE