MECHANISM: One-unit recurrent width boundary test

HYPOTHESIS: A 102-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 496,282,840 to approximately 487,474,320.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from the current 110 units to 102 while preserving the qualified batch-64 training regime and 16-frame schedule.

EVIDENCE: The adjacent 103-unit design qualified at 85.644% accuracy and 496,282,840 MACs, leaving a 0.644-point margin; 102 units is the next untested lower-cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
=======
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
>>>>>>> REPLACE