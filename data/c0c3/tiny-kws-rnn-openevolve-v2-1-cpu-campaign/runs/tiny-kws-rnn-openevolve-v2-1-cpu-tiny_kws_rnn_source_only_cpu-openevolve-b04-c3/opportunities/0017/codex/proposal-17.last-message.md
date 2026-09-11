MECHANISM: Dual-readout recurrent-width boundary search

HYPOTHESIS: A 126-unit GRU retaining the qualified 27-frame schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing total inference MACs from 1.234B to approximately 1.216B.

INTENDED_EDIT: Reduce the GRU state, temporal summary, and concatenated classifier input by one unit while preserving training and temporal coverage.

EVIDENCE: Adding terminal-state features raised the same 127-unit, 27-step schedule from 84.91% to 85.40%; the observed 128-to-127 width reduction at 28 steps cost only about 0.25 percentage points, suggesting the dual readout has enough margin for the next width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(254, 8)
=======
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
>>>>>>> REPLACE