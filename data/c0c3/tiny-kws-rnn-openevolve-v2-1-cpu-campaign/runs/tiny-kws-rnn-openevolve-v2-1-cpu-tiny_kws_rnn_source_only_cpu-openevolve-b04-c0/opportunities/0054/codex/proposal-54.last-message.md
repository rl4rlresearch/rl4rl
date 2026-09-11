MECHANISM: Boundary-aware temporal subsampling

HYPOTHESIS: A 68-unit GRU using the qualifying 24-frame schedule with only the earliest boundary frame removed will reach at least 85% accuracy in 23 steps and use fewer MACs than the verified 67-unit, 24-step model.

INTENDED_EDIT: Increase hidden width from 67 to 68 and replace uniform 23-frame resampling with the verified 24-frame sampling pattern minus frame zero.

EVIDENCE: Uniform 68-unit/23-step sampling achieved 83.93% accuracy but better cross-entropy than the qualifying 67-unit/24-step model; preserving 23 of that qualifying model’s exact sample locations tests whether schedule displacement, rather than insufficient capacity, caused the regression.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(268, 8)
=======
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(24, available_frames)
        return [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 23:
            return list(range(available_frames))
        qualifying_schedule = [
            step * (available_frames - 1) // 23
            for step in range(24)
        ]
        return qualifying_schedule[1:]
>>>>>>> REPLACE