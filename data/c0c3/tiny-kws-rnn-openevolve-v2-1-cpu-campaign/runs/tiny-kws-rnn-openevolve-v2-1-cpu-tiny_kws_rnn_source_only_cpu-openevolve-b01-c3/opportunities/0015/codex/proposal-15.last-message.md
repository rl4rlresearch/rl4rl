MECHANISM: Nonduplicate boundary-width trim

HYPOTHESIS: A 105-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 514,134,600.

INTENDED_EDIT: Reduce recurrent-state, temporal-summary, and classifier width from the current 116 units to 105 while preserving training and temporal coverage.

EVIDENCE: The 107-unit design achieved 86.135% accuracy with 532,299,320 MACs; because the subsequent 106-unit submission reproduced an already verified implementation, 105 units is the next untested lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 105, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(105, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 105, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
>>>>>>> REPLACE