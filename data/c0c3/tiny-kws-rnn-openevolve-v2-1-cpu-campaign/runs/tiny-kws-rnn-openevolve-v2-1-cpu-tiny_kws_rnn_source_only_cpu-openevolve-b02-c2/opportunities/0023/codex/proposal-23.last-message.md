MECHANISM: Adjacent recurrent-width refinement

HYPOTHESIS: A 93-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 761,043,740 to approximately 746,353,365.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 98 to 93 while preserving the qualified 29-frame schedule and training procedure.

EVIDENCE: The 94-unit, 29-step design achieved 87.36% accuracy with 761,043,740 MACs, outperforming wider qualified designs and leaving 2.36 percentage points of margin for the adjacent structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 93, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(186, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 93, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 93, device=device, dtype=dtype)
>>>>>>> REPLACE