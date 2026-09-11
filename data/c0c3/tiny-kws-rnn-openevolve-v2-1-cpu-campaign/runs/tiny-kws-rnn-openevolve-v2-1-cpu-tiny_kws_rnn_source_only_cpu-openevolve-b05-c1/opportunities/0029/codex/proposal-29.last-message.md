MECHANISM: Max-supported recurrent width compression

HYPOTHESIS: An 81-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.2%, from 533.30M to 521.65M.

INTENDED_EDIT: Reduce the GRU and recurrent summary width from 82 to 81 and resize the concatenated classifier input from 246 to 243, preserving the frame schedule and training procedure.

EVIDENCE: The 82-unit/26-step model achieved 86.50% accuracy, leaving 1.50 percentage points of margin, and every prior one-unit max-readout compression from 85 through 82 units passed; an earlier 81-unit model also passed at 31 steps.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
=======
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(243, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 82, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 81, device=device, dtype=dtype)
>>>>>>> REPLACE