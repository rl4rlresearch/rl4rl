MECHANISM: Two-unit recurrent-width refinement with full temporal coverage

HYPOTHESIS: A 58-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.9% versus the verified 60-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 60 to 58 and resize the classifier and recurrent state tensors accordingly.

EVIDENCE: The 60-unit full-sequence model achieved 86.50% accuracy, while the attempted 56-unit model produced no accuracy evidence because training timed out; testing the midpoint isolates a smaller, lower-risk structural reduction while retaining the successful full-frame dual readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
>>>>>>> REPLACE