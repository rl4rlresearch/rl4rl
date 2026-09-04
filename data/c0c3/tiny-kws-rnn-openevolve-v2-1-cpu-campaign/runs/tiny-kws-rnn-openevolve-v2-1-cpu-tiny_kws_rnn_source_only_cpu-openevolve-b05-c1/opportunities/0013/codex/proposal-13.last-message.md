MECHANISM: Incremental dual-timescale GRU width compression

HYPOTHESIS: An 80-unit GRU with all 32 frames and the mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.5% versus the verified 82-unit model.

INTENDED_EDIT: Reduce the GRU and summary width from 82 to 80 and resize the concatenated classifier input from 164 to 160, leaving training and temporal coverage unchanged.

EVIDENCE: The verified 82-unit model achieved 86.01% accuracy at 655.47M total MACs, and every tested dual-readout width from 82 through 91 passed, making the next two-unit reduction the most direct probe of the minimum viable width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
>>>>>>> REPLACE