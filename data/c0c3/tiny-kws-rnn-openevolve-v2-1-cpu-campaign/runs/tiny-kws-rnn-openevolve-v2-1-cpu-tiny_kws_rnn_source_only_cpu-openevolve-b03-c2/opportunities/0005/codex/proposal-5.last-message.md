MECHANISM: Moderate recurrent-width reduction

HYPOTHESIS: A 120-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the current design and below the qualified 29-step, 128-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 120 while preserving the full causal frame schedule and training procedure.

EVIDENCE: The current 128-unit, 32-step model achieves 87.24% accuracy with a 2.24-point margin, and recurrent computation accounts for virtually all MACs; a moderate width reduction targets that dominant cost while retaining all temporal evidence, unlike the failed 28-step schedule.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE