MECHANISM: Conservative full-resolution GRU width reduction

HYPOTHESIS: A 104-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 6.7% versus the qualified 108-unit design.

INTENDED_EDIT: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 128 to 104 while preserving the full-frame schedule and training procedure.

EVIDENCE: The 108-unit full-resolution GRU achieved 86.01% accuracy, while reducing temporal resolution to 24 or 16 steps failed below 83%; this supports preserving all frames and testing a small additional width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE