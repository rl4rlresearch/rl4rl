MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 88-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 689,009,965 to approximately 675,028,640.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 95 to 88 while preserving the qualified frame schedule and training procedure.

EVIDENCE: The 89-unit, 29-step design achieved 86.38% validation accuracy with 689,009,965 MACs, leaving 1.38 percentage points of margin and making width 88 the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 95, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(190, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 95, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE