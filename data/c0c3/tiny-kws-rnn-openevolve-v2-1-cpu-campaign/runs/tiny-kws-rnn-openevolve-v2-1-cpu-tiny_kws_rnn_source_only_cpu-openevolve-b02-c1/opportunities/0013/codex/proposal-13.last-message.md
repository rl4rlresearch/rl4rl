MECHANISM: Narrower two-block gated recurrence

HYPOTHESIS: Two independent 60-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified two-by-64 model.

INTENDED_EDIT: Reduce both parallel GRUs from 64 to 60 units, resize their states and concatenated classifier input from 128 to 120, and preserve the full-frame schedule and training procedure.

EVIDENCE: The two-by-64 full-frame model achieved 85.89% accuracy at 842,071,040 total MACs, while the three-by-43 attempt timed out; narrowing the successful two-module design tests its capacity boundary without adding recurrent-module overhead or discarding temporal frames.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE