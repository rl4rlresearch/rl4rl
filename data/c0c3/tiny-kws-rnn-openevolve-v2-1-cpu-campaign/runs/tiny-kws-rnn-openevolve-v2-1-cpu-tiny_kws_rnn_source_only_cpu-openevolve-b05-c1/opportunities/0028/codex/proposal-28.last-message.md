MECHANISM: Max-supported recurrent width compression

HYPOTHESIS: An 82-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 83-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary width from 83 to 82 and resize the classifier input from 249 to 246, preserving the frame schedule and training procedure.

EVIDENCE: The 83-unit/26-step model achieved 86.87% accuracy, giving 1.87 percentage points of margin, while an 82-unit model previously achieved 85.52% at 30 steps; this supports testing the smallest structural width reduction with the stronger max-augmented readout intact.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(249, 8)
=======
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 83, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 82, device=device, dtype=dtype)
>>>>>>> REPLACE