MECHANISM: Recurrent-width boundary search with full temporal coverage

HYPOTHESIS: A 100-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 104 to 100 while preserving all 32 causal steps and the training procedure.

EVIDENCE: The 104-unit full-frame model achieved 85.52% accuracy, and each prior width reduction from 120 through 108 to 104 remained feasible; another four-unit reduction directly probes the remaining width boundary without discarding temporal information.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE