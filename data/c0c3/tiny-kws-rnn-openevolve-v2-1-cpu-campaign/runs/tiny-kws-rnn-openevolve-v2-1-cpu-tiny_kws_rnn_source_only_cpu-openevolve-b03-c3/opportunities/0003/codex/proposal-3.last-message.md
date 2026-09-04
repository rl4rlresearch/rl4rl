MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 112-unit GRU processing all 32 consecutive frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit model.

INTENDED_EDIT: Reduce the GRU hidden width and classifier input width from 128 to 112 while preserving the successful full-frame schedule and training procedure.

EVIDENCE: The 128-unit, 32-step model achieved 87.24% accuracy, whereas temporal subsampling to either 24 or 16 frames fell below 83%; this indicates temporal resolution should be preserved and model width is the next structural cost dimension to test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE