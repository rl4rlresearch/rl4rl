MECHANISM: Endpoint-preserving causal subsampling with GRU width pruning

HYPOTHESIS: Processing 24 evenly spaced frames with 112 hidden units will retain at least 85% accuracy while reducing recurrent MACs by approximately 41.5%.

INTENDED_EDIT: Reduce the GRU and summary width from 128 to 112, and replace the 32-frame schedule with 24 uniformly distributed causal frame indices that retain both endpoints.

EVIDENCE: The current 128-unit, 32-step model reaches 87.24% accuracy, providing a 2.24-point margin while spending nearly all 1.483B inference MACs in recurrence; both temporal density and hidden width are therefore direct reduction targets.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(24, available_frames)
        if target_steps == available_frames:
            return list(range(available_frames))
        return [
            round(step * (available_frames - 1) / (target_steps - 1))
            for step in range(target_steps)
        ]
>>>>>>> REPLACE