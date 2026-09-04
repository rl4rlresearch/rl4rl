MECHANISM: Width-reduced full-resolution recurrence

HYPOTHESIS: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 22% versus the 128-unit baseline.

INTENDED_EDIT: Reduce the GRU state, temporal summary, and classifier input width from 128 to 112 without temporal subsampling.

EVIDENCE: The 128-unit, 32-step model achieved 87.24% accuracy, while reducing temporal coverage to 24 or 16 steps failed; preserving every frame while trimming recurrent width targets compute without discarding causal observations.

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