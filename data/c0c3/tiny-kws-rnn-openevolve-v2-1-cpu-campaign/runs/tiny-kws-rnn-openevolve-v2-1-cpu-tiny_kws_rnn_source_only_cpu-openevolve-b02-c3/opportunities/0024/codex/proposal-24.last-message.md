MECHANISM: Incremental temporal trimming of the qualified narrow GRU

HYPOTHESIS: An 88-unit dual-readout GRU processing the most recent 28 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 29-frame model’s 675,028,640.

INTENDED_EDIT: Reduce the GRU width from 94 to 88, resize its state and classifier, and omit the first four input frames.

EVIDENCE: The 88-unit model qualified at 86.38% over 30 frames and 86.50% over 29 frames; the latter leaves 1.50 percentage points of margin and motivates testing the next single-frame structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 30, 0)
=======
        start = max(available_frames - 28, 0)
>>>>>>> REPLACE