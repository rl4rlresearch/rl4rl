MECHANISM: Incremental span-preserving temporal thinning with causal deltas

HYPOTHESIS: A 74-unit delta-augmented GRU using 26 evenly distributed frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 557,867,500 to approximately 537,241,480.

INTENDED_EDIT: Install the qualified 74-unit recurrent width and reduce its schedule from 27 to 26 causally ordered, span-preserving frames.

EVIDENCE: The 74-unit delta GRU achieved 85.89% accuracy at 27 steps and 86.01% at 28 steps; the small loss from thinning and 0.89-point remaining margin motivate testing one fewer step, which saves more MACs than another one-unit width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(150, 8)
=======
        self.gru = nn.GRU(40, 74, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(148, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 74, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 74, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
        return list(range(start, available_frames))
=======
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