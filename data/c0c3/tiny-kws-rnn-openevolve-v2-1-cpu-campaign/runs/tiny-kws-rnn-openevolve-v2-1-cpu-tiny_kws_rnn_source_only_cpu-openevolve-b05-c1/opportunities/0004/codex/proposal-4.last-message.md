MECHANISM: Trade recurrent width for full temporal resolution

HYPOTHESIS: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 22% versus the verified 24-step, 128-unit design.

INTENDED_EDIT: Reduce the GRU, temporal summary, and classifier width from 128 to 96, and restore the complete 32-frame causal schedule.

EVIDENCE: The 128-unit model achieved 87.24% at 32 steps, while temporal subsampling to 16 and 20 steps failed and 24 steps only reached 85.77%; this suggests preserving temporal detail while reducing excess recurrent capacity is the better cost tradeoff.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(24, available_frames)
=======
        steps = min(32, available_frames)
>>>>>>> REPLACE