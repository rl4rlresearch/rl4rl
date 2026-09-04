MECHANISM: Full-resolution boundary-width reduction

HYPOTHESIS: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the qualified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 100 while preserving the proven full-frame schedule and training procedure.

EVIDENCE: The 104-unit model qualified at 85.89% accuracy, while the 96-unit model reached only 84.29%; testing the 100-unit midpoint is the most informative next probe of the width–accuracy boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE