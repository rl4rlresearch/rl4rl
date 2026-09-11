MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: A 90-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 717,398,045 to approximately 703,133,100.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 93 to 90 while preserving the qualified frame schedule and training procedure.

EVIDENCE: The 91-unit, 29-step design achieved 86.75% validation accuracy with 717,398,045 MACs, leaving 1.75 percentage points of margin and making width 90 the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 93, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(186, 8)
=======
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 93, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 93, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
>>>>>>> REPLACE