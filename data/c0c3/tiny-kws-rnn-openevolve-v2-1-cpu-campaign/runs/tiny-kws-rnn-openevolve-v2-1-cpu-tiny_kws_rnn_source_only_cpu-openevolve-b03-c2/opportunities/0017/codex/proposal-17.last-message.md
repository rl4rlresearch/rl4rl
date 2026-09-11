MECHANISM: Dual-view temporal readout with conservative recurrent-width reduction

HYPOTHESIS: A 110-unit dual-readout GRU on the qualified 26-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.7% versus the 111-unit design.

INTENDED_EDIT: Use the qualified 26-step nested schedule and concatenated mean/final-state readout, while reducing the GRU and summary width from 111 to 110.

EVIDENCE: The 111-unit dual-readout 26-step model achieved 86.13% accuracy; the observed 111-to-110 width reduction cost 0.86 percentage points at 30 steps, implying approximately 85.28% if that effect transfers.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(220, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 5:
            return [
                frame
                for frame in schedule
                if frame
                not in (1, available_frames - 3, available_frames - 2)
            ]
        return schedule
>>>>>>> REPLACE