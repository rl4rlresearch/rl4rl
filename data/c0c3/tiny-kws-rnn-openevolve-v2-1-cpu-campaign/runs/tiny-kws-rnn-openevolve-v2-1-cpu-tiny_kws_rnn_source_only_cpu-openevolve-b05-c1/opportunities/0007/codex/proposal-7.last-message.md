MECHANISM: Incremental dual-timescale GRU width compression

HYPOTHESIS: A 90-unit GRU retaining the mean-and-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.0% versus the verified 91-unit model.

INTENDED_EDIT: Reduce the GRU state and temporal-summary width from 91 to 90 and resize the concatenated classifier input from 182 to 180, preserving all 32 causal steps and the verified training procedure.

EVIDENCE: The 91-unit dual-readout model achieved 86.01% accuracy at 791.49M total MACs, giving a 1.01-point margin above the requirement and motivating the next incremental structural reduction without sacrificing temporal coverage.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(182, 8)
=======
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
>>>>>>> REPLACE