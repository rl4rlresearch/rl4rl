MECHANISM: Incremental dual-timescale GRU width compression

HYPOTHESIS: An 82-unit GRU with the full 32-frame schedule and mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.3% versus the verified 84-unit model.

INTENDED_EDIT: Reduce the GRU and summary width from 84 to 82 and resize the concatenated classifier input from 168 to 164, leaving training and temporal coverage unchanged.

EVIDENCE: The verified 84-unit model achieved 87.73% accuracy, 2.73 points above the requirement, and all tested dual-readout widths from 84 through 91 passed; this supports another cautious two-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
>>>>>>> REPLACE