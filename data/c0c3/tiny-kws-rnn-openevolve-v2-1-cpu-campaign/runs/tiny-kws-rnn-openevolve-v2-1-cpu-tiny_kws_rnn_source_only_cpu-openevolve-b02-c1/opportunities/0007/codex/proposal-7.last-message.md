MECHANISM: Boundary-search hidden-width reduction at full temporal resolution

HYPOTHESIS: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs below the verified 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 98 while preserving the successful training procedure and full-frame schedule.

EVIDENCE: The 100-unit model reached 85.40% accuracy while the 96-unit model reached 84.29%; testing the midpoint is the most informative probe of the minimum viable width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE