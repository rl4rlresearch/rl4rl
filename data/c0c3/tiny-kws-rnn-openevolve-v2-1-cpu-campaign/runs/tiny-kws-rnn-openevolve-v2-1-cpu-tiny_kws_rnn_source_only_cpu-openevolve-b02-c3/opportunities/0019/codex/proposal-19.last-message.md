MECHANISM: Dual-timescale recurrent width refinement

HYPOTHESIS: A 90-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below 756,998,080.

INTENDED_EDIT: Reduce the GRU hidden width from 92 to 90 and resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly.

EVIDENCE: The 92-unit, 30-frame model achieved 87.61% accuracy, improving on the qualified 94-unit model and leaving 2.61 percentage points of margin for the next two-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
=======
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
>>>>>>> REPLACE