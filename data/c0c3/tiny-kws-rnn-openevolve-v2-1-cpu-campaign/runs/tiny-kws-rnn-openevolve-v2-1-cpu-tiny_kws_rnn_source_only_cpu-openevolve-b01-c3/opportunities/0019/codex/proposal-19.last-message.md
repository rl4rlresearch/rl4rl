MECHANISM: One-unit recurrent width boundary test

HYPOTHESIS: A 101-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 487,474,320 to approximately 478,744,040.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 102 to 101 units while preserving the qualified batch-64 training regime and 16-frame schedule.

EVIDENCE: The adjacent 102-unit design achieved 85.276% accuracy at 487,474,320 MACs; 101 units is the next untested structural reduction, while the failed 15-frame result indicates temporal coverage should remain unchanged.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
=======
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
>>>>>>> REPLACE