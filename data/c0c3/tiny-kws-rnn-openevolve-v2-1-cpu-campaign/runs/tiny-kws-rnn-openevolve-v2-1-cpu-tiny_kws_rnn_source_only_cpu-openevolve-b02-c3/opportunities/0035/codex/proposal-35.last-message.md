MECHANISM: Single-unit refinement with span-preserving temporal thinning

HYPOTHESIS: A 73-unit delta-augmented GRU using 27 frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 557,867,500 to approximately 545,509,655.

INTENDED_EDIT: Reduce the GRU width from 76 to 73, resize its state and classifier, and adopt the qualified 27-step span-preserving schedule.

EVIDENCE: The 74-unit, 27-step model achieved 85.89% accuracy, while reducing width from 75 to 74 at 28 steps retained 86.01%; this supports testing the adjacent 73-unit boundary without sacrificing temporal coverage.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(152, 8)
=======
        self.gru = nn.GRU(40, 73, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(146, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 73, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]
>>>>>>> REPLACE