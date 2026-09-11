MECHANISM: Boundary-bisection GRU width compression

HYPOTHESIS: A 90-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.9% versus the verified 92-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 90 while retaining all 32 causal frames and the existing training procedure.

EVIDENCE: The 92-unit model achieved 85.89% accuracy, while the 88-unit model achieved 84.66%; width 90 directly probes the midpoint of the observed feasibility boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
=======
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(90, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
>>>>>>> REPLACE