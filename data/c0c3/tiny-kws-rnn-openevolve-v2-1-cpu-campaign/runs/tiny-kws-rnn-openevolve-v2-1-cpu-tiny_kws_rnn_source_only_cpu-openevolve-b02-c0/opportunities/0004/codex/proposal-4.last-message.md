MECHANISM: Incremental recurrent-width reduction

HYPOTHESIS: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving full causal temporal resolution and the existing training procedure.

EVIDENCE: The 112-unit full-resolution GRU achieved 86.13% accuracy, whereas reducing temporal coverage failed at both 24 and 16 steps; this motivates preserving all 32 frames and testing a conservative 8-unit width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE