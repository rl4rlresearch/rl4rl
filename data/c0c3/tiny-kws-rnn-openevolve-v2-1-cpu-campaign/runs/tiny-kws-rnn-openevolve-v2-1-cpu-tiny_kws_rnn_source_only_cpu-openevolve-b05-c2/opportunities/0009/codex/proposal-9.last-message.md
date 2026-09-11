MECHANISM: Boundary-seeking GRU width pruning with dual-timescale readout

HYPOTHESIS: An 84-unit full-resolution GRU will achieve at least 85% validation accuracy while reducing recurrent MACs by about 8% versus the qualified 88-unit design.

INTENDED_EDIT: Reduce the GRU and summary width from 88 to 84 and resize the mean/final classifier input from 176 to 168, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 88-unit model achieved 86.87% accuracy, while 80 units achieved 84.29%; 84 units tests the midpoint of the observed feasibility boundary, where interpolation suggests roughly 85.58% accuracy.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE