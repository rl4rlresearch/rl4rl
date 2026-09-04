MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 86-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 689,009,965 to approximately 647,491,420.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 89 to 86 while preserving the qualified schedule and training procedure.

EVIDENCE: The 87-unit, 29-step design achieved 86.01% validation accuracy with 661,189,125 MACs, leaving 1.01 percentage points of margin and making width 86 the closest unresolved structural reduction; reported prior 28-step reductions failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 89, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(178, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 89, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 89, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE