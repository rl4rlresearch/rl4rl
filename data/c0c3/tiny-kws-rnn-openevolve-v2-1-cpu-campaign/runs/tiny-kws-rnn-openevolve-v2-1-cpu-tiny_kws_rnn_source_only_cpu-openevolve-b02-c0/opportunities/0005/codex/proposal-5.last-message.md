MECHANISM: Incremental full-resolution GRU width reduction

HYPOTHESIS: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by roughly 14% versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state and classifier input width from 104 to 96 while preserving all frames and the existing training procedure.

EVIDENCE: Reducing width from 112 to 104 lowered accuracy by only 0.25 percentage points, from 86.13% to 85.89%, while reducing recurrent MACs from 1,156,700,160 to 1,008,983,040; this supports testing the next conservative eight-unit reduction without discarding temporal information.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE