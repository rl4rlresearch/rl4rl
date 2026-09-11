MECHANISM: Boundary-seeking width reduction at the qualified 28-step frontier

HYPOTHESIS: A 102-unit GRU processing frames 4–31 will retain at least 85% validation accuracy while reducing recurrent MACs from 867,319,740 to approximately 851,916,240.

INTENDED_EDIT: Use the best qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 103 to 102 units.

EVIDENCE: The 103-unit frames 4–31 design achieved 85.40% accuracy, while the 102-unit full-resolution model missed by only 0.09 percentage points; testing 102 units on the current cost frontier is the smallest structural reduction supported by the evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(2, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
>>>>>>> REPLACE