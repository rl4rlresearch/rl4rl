MECHANISM: Two-unit dual-timescale GRU width compression

HYPOTHESIS: An 86-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.1% versus the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU state and summary width from 88 to 86 and resize the concatenated classifier input from 176 to 172, leaving temporal coverage and training unchanged.

EVIDENCE: The 88-unit full-frame model achieved 86.87% accuracy—1.87 points above the requirement—at 744.74M total MACs, supporting another cautious two-unit structural reduction without revisiting temporal subsampling, which failed at 16 and 20 steps.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE