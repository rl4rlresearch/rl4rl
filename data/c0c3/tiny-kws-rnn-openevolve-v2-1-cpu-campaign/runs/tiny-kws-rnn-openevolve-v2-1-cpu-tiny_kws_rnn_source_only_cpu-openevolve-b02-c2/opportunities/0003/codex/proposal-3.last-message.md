MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 120-unit GRU using all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the qualified 128-unit baseline.

INTENDED_EDIT: Reduce GRU state and classifier width from 128 to 120 without changing temporal sampling or training.

EVIDENCE: The 128-unit, 32-step model achieved 87.24% accuracy, while both 24- and 28-step variants failed; preserving every frame tests whether temporal information, rather than modest excess hidden capacity, caused those regressions.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE