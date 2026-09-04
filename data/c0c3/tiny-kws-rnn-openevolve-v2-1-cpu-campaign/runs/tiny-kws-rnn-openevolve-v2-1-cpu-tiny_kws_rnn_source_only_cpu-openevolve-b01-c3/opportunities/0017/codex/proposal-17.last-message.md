MECHANISM: One-unit recurrent width boundary test

HYPOTHESIS: A 103-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 505,169,600 to approximately 496,282,840.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 107 to 103 units while preserving the qualified batch-64 training regime and 16-frame schedule.

EVIDENCE: The adjacent 104-unit design achieved 86.626% accuracy at 505,169,600 MACs, leaving a 1.626-point margin and motivating the smallest untested structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 107, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(107, 8)
=======
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 107, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 107, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE