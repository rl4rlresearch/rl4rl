MECHANISM: Full-resolution GRU width pruning with dual-timescale readout

HYPOTHESIS: A 96-unit GRU using all 32 frames and concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13.6% versus the qualified 104-unit design.

INTENDED_EDIT: Reduce the recurrent and summary width from 104 to 96 and resize the dual-readout classifier accordingly, while preserving the complete causal frame schedule and training procedure.

EVIDENCE: The 104-unit full-resolution dual-readout model achieved 88.34% accuracy, improving on the 111-unit model’s 87.85% despite lower width; meanwhile both 24-frame variants failed, motivating another moderate width reduction without temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE