MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 83-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 620,521,440 to approximately 607,249,165.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 84 to 83 while preserving the qualified schedule and training procedure.

EVIDENCE: The 84-unit design achieved 85.64% validation accuracy with 620,521,440 MACs; its positive 0.64-point margin makes width 83 the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(166, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
>>>>>>> REPLACE