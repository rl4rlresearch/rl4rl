MECHANISM: Conservative full-resolution recurrent width reduction

HYPOTHESIS: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.5% versus the qualified 112-unit design.

INTENDED_EDIT: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 128 to 108 while preserving the successful full-frame schedule and training procedure.

EVIDENCE: The 112-unit full-resolution GRU achieved 85.77% accuracy, whereas reducing temporal resolution caused both 24-step and 16-step models to fall below 83%; a modest further width reduction is therefore the lowest-risk structural cost reduction supported by prior results.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE