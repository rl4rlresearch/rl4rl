MECHANISM: Full-resolution recurrent width bottleneck

HYPOTHESIS: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12.8% versus the qualified 112-unit model, from 1,156,700,160 to approximately 1,008,983,040.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 104 while preserving complete temporal coverage and the existing training procedure.

EVIDENCE: The full-resolution 112-unit GRU achieved 86.26% accuracy at substantially lower cost than the 128-unit baseline, while all tested temporal subsampling variants failed; this supports another conservative width reduction without dropping frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE