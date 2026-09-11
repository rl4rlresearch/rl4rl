MECHANISM: Fine-grained recurrent-width scaling with full temporal coverage

HYPOTHESIS: A 56-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9.7% versus the verified 60-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 60 to 56 and resize the classifier and recurrent state tensors accordingly.

EVIDENCE: The 60-unit full-sequence model achieved 86.50% accuracy, and every tested width reduction from 104 through 60 units remained above the target; another four-unit reduction preserves the proven temporal coverage and dual mean-plus-final readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 56, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 56, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 56, device=device, dtype=dtype)
>>>>>>> REPLACE