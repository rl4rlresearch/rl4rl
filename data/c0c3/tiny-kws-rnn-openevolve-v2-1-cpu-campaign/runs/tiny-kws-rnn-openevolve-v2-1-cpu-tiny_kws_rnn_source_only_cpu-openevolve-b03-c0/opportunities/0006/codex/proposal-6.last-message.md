MECHANISM: Recurrent-width boundary search with full temporal coverage

HYPOTHESIS: A 104-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the verified 108-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 108 to 104 while preserving all 32 causal steps and the verified training procedure.

EVIDENCE: The 108-unit full-frame model achieved 86.01% accuracy, slightly exceeding the 112-unit model’s 85.77%, so another modest four-unit reduction is the most direct test of the feasible width boundary without discarding temporal information.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE