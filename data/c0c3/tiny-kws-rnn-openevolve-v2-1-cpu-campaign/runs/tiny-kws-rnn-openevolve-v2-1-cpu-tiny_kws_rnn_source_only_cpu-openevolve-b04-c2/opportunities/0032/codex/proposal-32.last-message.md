MECHANISM: Width-for-temporal-context exchange

HYPOTHESIS: A 97-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference cost to approximately 611.1 million MACs.

INTENDED_EDIT: Replace the 103-unit, 20-step frontier with a narrower 97-unit GRU processing 22 contiguous frames, using batch size 16 for additional optimizer updates.

EVIDENCE: The 103-unit 22-step design reached 85.89% accuracy, providing substantially more margin than the 20-step design’s 85.03%; reducing width to 97 lowers estimated recurrent MACs despite restoring two informative frames.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE