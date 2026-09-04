MECHANISM: Boundary-search hidden-width reduction at full temporal resolution

HYPOTHESIS: A 100-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 100, preserving the successful full-frame schedule and training procedure.

EVIDENCE: The 104-unit model achieved 85.89% accuracy while the 96-unit model achieved 84.29%; testing the midpoint is the most informative next probe of the minimum viable width.

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