MECHANISM: One-unit recurrent width boundary trim

HYPOTHESIS: A 99-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 470,092,000 to approximately 461,518,200.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 units while preserving the qualified training procedure and 16-frame schedule.

EVIDENCE: The adjacent 100-unit design achieved 85.521% accuracy at 470,092,000 MACs; its 0.521-point margin makes 99 units the smallest untested lower-cost boundary, while the failed 15-frame experiment supports keeping temporal coverage unchanged.

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