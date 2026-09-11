MECHANISM: Fine-grained recurrent-width boundary search

HYPOTHESIS: A 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.8% versus the verified 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 100 to 99 while preserving all 32 causal steps and the verified training procedure.

EVIDENCE: The 100-unit full-frame GRU achieved 85.03% accuracy, while all tested widths from 104 through 120 also passed; because the remaining accuracy margin is narrow, a one-unit reduction is the most informative low-risk refinement of the feasible width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE