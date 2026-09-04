MECHANISM: Two-unit dual-timescale GRU width compression

HYPOTHESIS: An 88-unit GRU retaining the full 32-frame schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 4% versus the verified 90-unit model.

INTENDED_EDIT: Reduce the GRU state and summary width from 90 to 88 and resize the concatenated classifier input from 180 to 176, leaving training and temporal coverage unchanged.

EVIDENCE: The verified 90-unit model achieved 87.12% accuracy, 2.12 points above the requirement, at 775.75M total MACs; this unusually strong margin supports testing a two-unit structural reduction while preserving the readout and all 32 frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE