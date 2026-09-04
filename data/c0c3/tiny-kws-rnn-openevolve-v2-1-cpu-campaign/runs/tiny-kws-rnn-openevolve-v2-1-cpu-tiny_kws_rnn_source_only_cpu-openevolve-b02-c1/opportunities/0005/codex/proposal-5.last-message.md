MECHANISM: Hidden-width reduction at full temporal resolution

HYPOTHESIS: A 96-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 96 while preserving the successful full-frame schedule and training procedure.

EVIDENCE: Reducing width from 112 to 104 lowered accuracy by only 0.25 percentage points, from 86.13% to 85.89%, while temporal subsampling failed; another eight-unit reduction directly extends the successful width-efficiency trend without discarding frames.

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