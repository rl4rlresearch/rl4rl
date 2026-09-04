MECHANISM: Width-for-temporal-resolution tradeoff

HYPOTHESIS: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.8% versus the verified 128-unit, 28-step model.

INTENDED_EDIT: Reduce the GRU and classifier width from 128 to 112 units, resize recurrent state tensors accordingly, and restore the complete 32-frame causal schedule.

EVIDENCE: The 128-unit model achieved 87.24% with 32 frames, while reducing temporal coverage to 27 frames fell to 84.05%; restoring all frames provides accuracy headroom while structural width reduction lowers dense MACs.

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
        steps = min(28, available_frames)
=======
        steps = min(32, available_frames)
>>>>>>> REPLACE