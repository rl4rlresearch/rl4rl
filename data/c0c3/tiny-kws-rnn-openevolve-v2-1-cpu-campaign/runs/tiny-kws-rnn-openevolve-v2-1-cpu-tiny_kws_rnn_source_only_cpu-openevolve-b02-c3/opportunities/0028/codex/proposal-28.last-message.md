MECHANISM: Span-preserving temporal thinning

HYPOTHESIS: An 88-unit dual-readout GRU using 26 evenly thinned frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 628,554,080 to approximately 605,316,800.

INTENDED_EDIT: Replace the 128-unit baseline with the qualified 88-unit mean-plus-final GRU and process 26 samples spanning the latest 28 frames instead of truncating the window to 26 contiguous frames.

EVIDENCE: The 88-unit model achieved 85.89% on 28 frames and 85.03% on 27, whereas reducing width to 86 missed qualification; retaining width and removing two distributed, temporally redundant frames tests a lower-cost alternative while preserving the stronger 28-frame temporal span.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
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
        target_steps = min(26, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]
>>>>>>> REPLACE