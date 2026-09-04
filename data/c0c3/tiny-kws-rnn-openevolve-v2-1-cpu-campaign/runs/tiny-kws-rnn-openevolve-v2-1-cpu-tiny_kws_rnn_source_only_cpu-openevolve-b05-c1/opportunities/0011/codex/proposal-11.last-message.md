MECHANISM: Incremental dual-timescale GRU width compression

HYPOTHESIS: An 84-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.2% versus the verified 86-unit model.

INTENDED_EDIT: Reduce the GRU and summary width from 86 to 84 and resize the concatenated classifier input from 172 to 168, leaving temporal coverage and training unchanged.

EVIDENCE: The verified 86-unit model achieved 86.63% accuracy, 1.63 points above the requirement, while every tested dual-readout width from 86 through 91 passed; this supports another cautious two-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE