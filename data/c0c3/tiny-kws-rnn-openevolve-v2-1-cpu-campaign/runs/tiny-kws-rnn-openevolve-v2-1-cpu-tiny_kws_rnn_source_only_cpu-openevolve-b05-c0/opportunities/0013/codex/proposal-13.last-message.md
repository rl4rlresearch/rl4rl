MECHANISM: Fine-grained recurrent-width scaling with full temporal coverage

HYPOTHESIS: A 64-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9% versus the verified 68-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 68 to 64 and resize the classifier and recurrent state tensors accordingly.

EVIDENCE: The 68-unit full-sequence model achieved 86.87% accuracy, and every tested width reduction from 104 through 68 units remained viable; preserving all frames while testing the next four-unit reduction is the best-supported lower-cost change.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(136, 8)
=======
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
>>>>>>> REPLACE