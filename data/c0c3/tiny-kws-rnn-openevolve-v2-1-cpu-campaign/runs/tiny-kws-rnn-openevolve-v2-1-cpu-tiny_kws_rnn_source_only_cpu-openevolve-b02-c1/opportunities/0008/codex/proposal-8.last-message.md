MECHANISM: One-unit boundary-search width reduction

HYPOTHESIS: A 99-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs below the verified 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 99 while preserving the full-frame schedule and training procedure.

EVIDENCE: The 100-unit model met the target at 85.40% accuracy, while the 98-unit model missed it at 84.54%; width 99 is the only untested integer at the observed feasibility boundary.

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