MECHANISM: Incremental edge-frame trimming with dual-timescale readout

HYPOTHESIS: A 98-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 31-frame design.

INTENDED_EDIT: Replace the 99-unit mean-only model with the qualified 98-unit mean-plus-final readout and omit the first two input frames.

EVIDENCE: The 98-unit dual-readout model achieved 87.12% over 32 frames and 86.50% over the most recent 31 frames; its remaining 1.50-point margin motivates testing one additional edge-frame reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 30, 0)
        return list(range(start, available_frames))
>>>>>>> REPLACE