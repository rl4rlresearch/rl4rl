MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 87-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 675,028,640 to approximately 661,189,125.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 88 to 87 while preserving the qualified frame schedule and training procedure.

EVIDENCE: The 88-unit, 29-step model achieved 86.26% validation accuracy with 675,028,640 MACs, leaving 1.26 percentage points of margin and making width 87 the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 87, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 87, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 87, device=device, dtype=dtype)
>>>>>>> REPLACE