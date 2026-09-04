MECHANISM: Max-supported recurrent width compression

HYPOTHESIS: An 84-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 569.02M to 556.99M.

INTENDED_EDIT: Reduce the GRU and all summary-state widths from 85 to 84 and resize the concatenated classifier input from 255 to 252, preserving the verified schedule and training procedure.

EVIDENCE: The 85-unit/26-step max-readout model achieved 86.26%, while an 84-unit model without max pooling passed at 85.64% over 28 steps; this supports testing one-unit compression while retaining the stronger readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(255, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 85, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE