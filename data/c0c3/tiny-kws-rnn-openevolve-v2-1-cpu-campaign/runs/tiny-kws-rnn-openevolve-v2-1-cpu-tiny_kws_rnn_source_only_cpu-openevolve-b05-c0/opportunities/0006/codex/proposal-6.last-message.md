MECHANISM: Rebalance recurrent width for complete temporal coverage

HYPOTHESIS: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 10.8% versus the verified 128-unit, 28-frame model.

INTENDED_EDIT: Reduce GRU and classifier width from 128 to 112, resize recurrent state tensors accordingly, and restore the full causal frame sequence.

EVIDENCE: The 128-unit model reached 87.24% with all 32 frames, while reducing temporal coverage to 26–27 frames caused threshold failures; full coverage paired with a modest width reduction tests whether temporal evidence is more valuable than excess hidden capacity at lower MAC cost.

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
        steps = min(28, available_frames)
        return [
            index * (available_frames - 1) // (steps - 1)
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
>>>>>>> REPLACE