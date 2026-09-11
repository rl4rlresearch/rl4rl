MECHANISM: One-unit recurrent width refinement at the qualified 29-step schedule

HYPOTHESIS: A 95-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 790,849,920 to approximately 775,875,925.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 96 to 95 while preserving the qualified schedule and training procedure.

EVIDENCE: The 96-unit 29-step model achieved 86.50% accuracy, matching the 97-unit model despite lower cost; its 1.50-point margin makes the adjacent 95-unit width the closest informative structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
        self.gru = nn.GRU(20, 95, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(190, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 95, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
>>>>>>> REPLACE