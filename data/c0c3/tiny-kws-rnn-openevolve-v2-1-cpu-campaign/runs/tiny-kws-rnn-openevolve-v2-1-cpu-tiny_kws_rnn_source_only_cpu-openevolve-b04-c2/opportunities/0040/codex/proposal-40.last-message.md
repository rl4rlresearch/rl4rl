MECHANISM: Late-context-preserving leading-frame trim

HYPOTHESIS: An 86-unit GRU processing frames 5–24 will reach at least 85% validation accuracy while retaining the 20-step cost of approximately 446,333,120 MACs.

INTENDED_EDIT: Use batch size 16, reduce the GRU to 86 units, and replace the unsuccessful frames 4–23 schedule with 20 contiguous frames shifted one step later.

EVIDENCE: The 86-unit frames 4–24 model qualified at 85.15%, while removing frame 24 missed qualification by only one example; the failed 87-unit rebound suggests preserving the final frame is more informative than adding capacity.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(86, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(5, available_frames - 7))
>>>>>>> REPLACE