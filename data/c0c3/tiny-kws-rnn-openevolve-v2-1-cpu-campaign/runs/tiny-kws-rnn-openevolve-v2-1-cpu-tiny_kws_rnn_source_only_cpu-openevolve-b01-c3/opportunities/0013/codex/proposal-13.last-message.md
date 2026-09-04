MECHANISM: One-unit boundary width trim

HYPOTHESIS: A 107-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 541,499,040 to approximately 532,299,320.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and classifier width from 108 to 107 units without changing training or temporal coverage.

EVIDENCE: The 108-unit design qualified at 85.399% accuracy, and the 110-unit design qualified at 86.012%; a one-unit reduction is the smallest informative capacity test below the current boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
=======
        self.gru = nn.GRU(20, 107, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(107, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 107, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 107, device=device, dtype=dtype)
>>>>>>> REPLACE