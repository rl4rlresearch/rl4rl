MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 89-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 703,133,100 to approximately 689,009,965.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 94 to 89 while preserving the qualified schedule and training procedure.

EVIDENCE: The 90-unit, 29-step design achieved 86.87% validation accuracy with 703,133,100 MACs, leaving 1.87 percentage points of margin and motivating the adjacent untested width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
=======
        self.gru = nn.GRU(20, 89, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(178, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 89, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 89, device=device, dtype=dtype)
>>>>>>> REPLACE