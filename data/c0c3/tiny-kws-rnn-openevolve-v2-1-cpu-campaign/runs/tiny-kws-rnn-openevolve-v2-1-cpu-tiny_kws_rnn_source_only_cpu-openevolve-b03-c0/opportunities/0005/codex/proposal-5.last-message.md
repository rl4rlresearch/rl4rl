MECHANISM: Recurrent-width boundary search with full temporal coverage

HYPOTHESIS: A 108-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.5% versus the verified 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 112 to 108 while preserving all 32 causal steps and the training procedure.

EVIDENCE: The 112-unit full-frame model achieved 85.77% accuracy after the 120-unit model achieved 86.75%; this supports testing 108 units as the next likely feasible point before the observed accuracy margin is exhausted.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE