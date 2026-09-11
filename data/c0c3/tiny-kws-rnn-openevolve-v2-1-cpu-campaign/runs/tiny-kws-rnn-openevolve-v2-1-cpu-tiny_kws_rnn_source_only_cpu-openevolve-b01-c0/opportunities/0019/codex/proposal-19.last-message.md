MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 125-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.47% versus the passing 126-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 126 to 125 while preserving the successful frame schedule, additive readout, and training procedure.

EVIDENCE: The 126-unit additive-readout model achieved 85.890% accuracy after the 127-unit version achieved 86.380%; both passed with identical temporal coverage, supporting the next isolated one-channel reduction. The earlier 125-unit result used a weaker mean-only readout and missed by just 0.092 percentage points, so it does not rule out 125 units with the stronger verified additive readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
>>>>>>> REPLACE