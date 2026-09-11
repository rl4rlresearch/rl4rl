MECHANISM: Recurrent-width compression with full temporal coverage

HYPOTHESIS: Reducing the GRU width from 128 to 120 will preserve at least 85% validation accuracy while lowering recurrent MACs by approximately 11.3%, because the failed 24- and 28-step trials indicate that retaining all 32 temporal observations is more valuable than retaining the full hidden width.

INTENDED_EDIT: Use a 120-unit GRU and matching classifier/state tensors while keeping the complete 32-frame schedule and training procedure unchanged.

EVIDENCE: The 128-unit, 32-step model reached 87.24% accuracy, whereas temporal subsampling fell below 85% at both 28 and 24 steps; this motivates compressing state capacity instead of discarding frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE