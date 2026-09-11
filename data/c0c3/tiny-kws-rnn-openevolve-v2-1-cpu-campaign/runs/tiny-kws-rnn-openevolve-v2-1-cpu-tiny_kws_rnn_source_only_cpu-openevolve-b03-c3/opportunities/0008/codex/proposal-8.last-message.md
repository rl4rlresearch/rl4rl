MECHANISM: One-unit full-resolution GRU width reduction

HYPOTHESIS: A 97-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 1.9% versus the qualified 98-unit design.

INTENDED_EDIT: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 112 to 97 while preserving the full-frame schedule and training procedure.

EVIDENCE: The 98-unit full-resolution GRU qualified at 85.03% accuracy with 905,406,320 MACs; because this is the observed capacity boundary, a one-unit reduction is the most precise next probe while retaining every frame.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE