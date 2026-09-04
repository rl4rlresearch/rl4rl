MECHANISM: Nested endpoint-preserving frame pruning

HYPOTHESIS: An 85-unit GRU using 27 frames formed by removing only the second frame from the successful 28-frame schedule will achieve at least 85% accuracy while reducing total inference MACs by approximately 1.5%.

INTENDED_EDIT: Increase the GRU width from 84 to 85 and replace uniform 27-frame resampling with a nested schedule that preserves 27 of the verified model’s 28 frame locations.

EVIDENCE: The 84-unit/28-step model passed at 85.64%, while the uniformly resampled 85-unit/27-step model narrowly failed at 84.54% despite lower MACs; that resampling changed seven frame locations, so pruning one likely low-information leading frame isolates step reduction from broad temporal realignment.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1]
        return indices
>>>>>>> REPLACE