MECHANISM: Joint recurrent-width reduction with qualified 30-frame subsampling

HYPOTHESIS: A 112-unit GRU processing 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 12% versus the qualified 120-unit, 30-step design.

INTENDED_EDIT: Reduce the GRU, summary state, and classifier width from 128 to 112, and use the qualified 30-frame causal schedule.

EVIDENCE: The 120-unit, 30-step model achieved 85.77% accuracy, while reducing width from 128 to 120 cost only 0.49 points at 32 steps; this supports testing the next 8-unit structural reduction while retaining the successful 30-frame schedule.

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

<<<<<<< SEARCH
        steps = min(29, available_frames)
=======
        steps = min(30, available_frames)
>>>>>>> REPLACE