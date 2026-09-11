MECHANISM: Conservative full-resolution recurrent width reduction

HYPOTHESIS: A 102-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.5% versus the qualified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 102 while preserving complete temporal coverage and the verified training procedure.

EVIDENCE: The 104-unit full-resolution GRU achieved 85.03% accuracy at 1,008,983,040 recurrent MACs, while every temporal-subsampling design failed; this motivates a conservative two-unit width reduction without dropping frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
>>>>>>> REPLACE