MECHANISM: Incremental edge-frame trimming with dual-timescale readout

HYPOTHESIS: A 98-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 30-frame design.

INTENDED_EDIT: Replace the current 100-unit mean-only GRU with the qualified 98-unit mean-plus-final architecture and omit the first three input frames.

EVIDENCE: The same 98-unit dual-readout architecture passed at 87.12% with 32 frames, 86.50% with 31 frames, and 85.77% with 30 frames; its remaining 0.77-point margin supports testing one further edge-frame reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
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
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 29, 0)
        return list(range(start, available_frames))
>>>>>>> REPLACE