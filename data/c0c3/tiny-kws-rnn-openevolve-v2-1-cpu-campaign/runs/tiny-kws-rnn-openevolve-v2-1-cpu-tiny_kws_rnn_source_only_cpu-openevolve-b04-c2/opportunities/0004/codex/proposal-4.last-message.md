MECHANISM: Full-resolution recurrent width bottleneck

HYPOTHESIS: A 112-unit GRU processing all 32 frames will preserve at least 85% validation accuracy while reducing recurrent MACs by 22.0%, from 1,482,178,560 to 1,156,700,160.

INTENDED_EDIT: Reduce the GRU state, temporal summary, and classifier input width from 128 to 112 while retaining complete temporal coverage.

EVIDENCE: The 128-unit, 32-step model achieved 87.24% accuracy, whereas every tested temporal subsampling schedule failed; this motivates preserving all frames and reducing recurrent width instead.

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