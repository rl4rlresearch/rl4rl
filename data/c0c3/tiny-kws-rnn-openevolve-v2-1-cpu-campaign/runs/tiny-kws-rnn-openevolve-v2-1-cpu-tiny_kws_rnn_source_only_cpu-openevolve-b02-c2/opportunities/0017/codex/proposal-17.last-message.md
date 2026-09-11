MECHANISM: Dual-view pooling with two-edge-frame omission

HYPOTHESIS: A 98-unit dual-view GRU processing frames 1–30 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 849,497,320 and recurrent steps to 24,450.

INTENDED_EDIT: Use a 98-unit GRU, classify concatenated mean and terminal recurrent states, and omit the first and final input frames.

EVIDENCE: The 98-unit dual-view model achieved 86.50% accuracy after omitting the first frame, leaving 1.50 percentage points of margin for testing one additional edge-frame omission; this should beat the qualified 97-unit, 31-step model’s 861,462,335 MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames - 1))
>>>>>>> REPLACE