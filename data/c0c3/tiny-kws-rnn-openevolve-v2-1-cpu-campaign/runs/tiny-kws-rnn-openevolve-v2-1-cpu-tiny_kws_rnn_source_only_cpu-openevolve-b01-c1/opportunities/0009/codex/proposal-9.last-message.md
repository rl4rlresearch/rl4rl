MECHANISM: Boundary-bisection GRU width compression

HYPOTHESIS: A 91-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs versus the verified 92-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 91 while preserving all 32 causal frames and the existing training procedure.

EVIDENCE: The 92-unit model achieved 85.89% accuracy, while widths 90 and 88 missed the threshold; width 91 is the only untested integer at the observed feasibility boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
=======
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
>>>>>>> REPLACE