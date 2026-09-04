MECHANISM: Boundary-seeking full-resolution GRU width compression

HYPOTHESIS: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the verified 96-unit design.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 96 to 92 while preserving the full causal frame schedule and training procedure.

EVIDENCE: The 96-unit full-resolution GRU achieved 85.15% accuracy after the 104-unit model achieved 86.01%; testing the midpoint width of 92 is the most direct probe of the feasible compression boundary without discarding temporally valuable frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
>>>>>>> REPLACE