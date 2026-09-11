MECHANISM: Fine-grained recurrent-width boundary search

HYPOTHESIS: A 98-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.84% versus the verified 99-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 99 to 98 while preserving all 32 causal steps and the verified training procedure.

EVIDENCE: The 99-unit full-frame GRU achieved 85.03% accuracy, identical to the 100-unit result, so a one-unit reduction is the most informative next probe of the feasible width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE