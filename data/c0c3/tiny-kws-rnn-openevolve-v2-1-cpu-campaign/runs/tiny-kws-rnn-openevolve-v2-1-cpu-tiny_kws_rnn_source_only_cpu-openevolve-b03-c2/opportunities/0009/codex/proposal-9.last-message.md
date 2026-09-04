MECHANISM: Single-unit recurrent-width boundary refinement

HYPOTHESIS: A 111-unit GRU on the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.64% versus the 112-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 111 without changing the training procedure or frame schedule.

EVIDENCE: The 112-unit model qualified at 85.276% accuracy, while 110 units achieved 84.540%; testing 111 units directly resolves the remaining width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE