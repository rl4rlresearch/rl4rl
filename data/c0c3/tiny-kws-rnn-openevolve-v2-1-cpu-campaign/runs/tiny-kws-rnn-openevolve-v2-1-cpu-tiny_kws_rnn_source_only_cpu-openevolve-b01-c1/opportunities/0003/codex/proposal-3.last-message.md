MECHANISM: Full-resolution recurrent width compression

HYPOTHESIS: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 128 to 112 while preserving the full causal frame schedule.

EVIDENCE: Temporal subsampling reduced accuracy from 87.24% to 83.19% even at 24 steps, indicating that all 32 frames are valuable; compressing hidden width instead tests an independent structural cost reduction without discarding temporal evidence.

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