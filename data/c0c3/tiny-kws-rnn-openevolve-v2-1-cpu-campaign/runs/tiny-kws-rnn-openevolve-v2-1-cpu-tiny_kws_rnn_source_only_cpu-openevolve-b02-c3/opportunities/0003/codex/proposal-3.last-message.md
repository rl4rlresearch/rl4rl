MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 41% versus the qualified 128-unit design.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 128 to 96 while preserving the full causal frame schedule.

EVIDENCE: Temporal subsampling to 16 and 24 steps failed at 82.58% and 83.44%, showing that full temporal coverage is important; the qualified 32-step model has 87.24% accuracy, providing margin to test structural width reduction instead.

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