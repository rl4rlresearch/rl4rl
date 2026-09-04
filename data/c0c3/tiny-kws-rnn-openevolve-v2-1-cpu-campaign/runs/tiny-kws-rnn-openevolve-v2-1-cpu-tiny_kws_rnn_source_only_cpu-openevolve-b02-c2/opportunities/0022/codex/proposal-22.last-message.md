MECHANISM: Adjacent recurrent-width refinement

HYPOTHESIS: A 94-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 775,875,925 to approximately 761,043,740.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from the current 97 units to 94 while preserving the qualified 29-frame schedule and training procedure.

EVIDENCE: The 95-unit, 29-step design achieved 86.87% accuracy with 775,875,925 MACs, improving slightly over the qualified 96- and 97-unit designs; its 1.87-point margin motivates testing the adjacent 94-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
>>>>>>> REPLACE