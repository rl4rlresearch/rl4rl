MECHANISM: Adjacent recurrent-width refinement

HYPOTHESIS: A 92-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 746,353,365 to approximately 731,804,800.

INTENDED_EDIT: Replace the current 128-unit mean-pooled GRU with a 92-unit GRU using concatenated mean and terminal states, and execute the qualified 29-frame schedule.

EVIDENCE: The 93-unit, 29-step dual-view design achieved 86.13% accuracy with 746,353,365 MACs, leaving 1.13 percentage points of margin and making the adjacent 92-unit width the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
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
        return list(range(1, available_frames - 2))
>>>>>>> REPLACE