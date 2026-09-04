MECHANISM: Capacity-for-time exchange with mild uniform subsampling

HYPOTHESIS: A 99-unit GRU processing 31 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 1.3% versus the verified 98-unit, 32-step model.

INTENDED_EDIT: Increase GRU width from 98 to 99 and use 31 unique increasing frame indices spanning the complete recording.

EVIDENCE: The 98- and 99-unit full-frame models both achieved 85.03%, while 97 units failed; restoring one unit provides the smallest evidenced capacity buffer for a much milder temporal reduction than the unsuccessful 24- and 28-frame trials.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 31:
            return list(range(available_frames))
        return [
            round(i * (available_frames - 1) / 30)
            for i in range(31)
        ]
>>>>>>> REPLACE