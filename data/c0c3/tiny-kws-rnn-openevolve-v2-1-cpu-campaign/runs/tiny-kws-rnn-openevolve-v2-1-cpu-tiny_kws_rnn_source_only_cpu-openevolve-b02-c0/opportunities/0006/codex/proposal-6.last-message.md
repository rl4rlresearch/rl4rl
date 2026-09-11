MECHANISM: Boundary-seeking recurrent-width reduction

HYPOTHESIS: A 100-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 100 while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 104-unit model achieved 85.89% accuracy, while the 96-unit model achieved 84.29%; testing the midpoint directly probes the narrowest width likely to satisfy the threshold without confounding temporal or training changes.

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