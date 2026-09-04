MECHANISM: Full-resolution midpoint width reduction

HYPOTHESIS: A 112-unit GRU processing all 32 frames will exceed 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the qualified 128-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 128 to 112 without changing temporal coverage or training.

EVIDENCE: The 128-unit full-resolution model achieved 87.24%, while the 96-unit model achieved 84.29%; 112 is the midpoint that directly brackets the smallest viable full-resolution width after reduced frame schedules failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE