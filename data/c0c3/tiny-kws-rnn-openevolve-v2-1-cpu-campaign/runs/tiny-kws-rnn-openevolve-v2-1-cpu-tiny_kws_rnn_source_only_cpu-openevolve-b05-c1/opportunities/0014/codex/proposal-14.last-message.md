MECHANISM: One-unit dual-timescale GRU width compression

HYPOTHESIS: An 81-unit GRU using all 32 frames and the mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the verified 82-unit model.

INTENDED_EDIT: Reduce the GRU state and summary width from 82 to 81 and resize the concatenated classifier input from 164 to 162, preserving temporal coverage and training.

EVIDENCE: The 82-unit model passed at 86.01% while the 80-unit model narrowly failed at 84.29%; testing 81 units directly resolves the remaining minimum-width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)
=======
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(162, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
>>>>>>> REPLACE