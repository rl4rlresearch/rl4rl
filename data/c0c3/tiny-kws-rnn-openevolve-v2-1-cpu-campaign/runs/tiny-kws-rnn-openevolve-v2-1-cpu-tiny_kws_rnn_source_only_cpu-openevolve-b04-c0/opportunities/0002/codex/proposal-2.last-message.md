MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 112-unit GRU will retain at least 85% accuracy while reducing recurrent MACs by about 22%, because it preserves all 32 frames whose decimation caused accuracy to fall to 81.72%.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier input width from 128 to 112 while retaining the full frame schedule.

EVIDENCE: The 128-unit, 32-step model achieved 87.24%, whereas reducing temporal coverage to 16 steps achieved only 81.72%; this motivates reducing recurrent capacity instead of discarding frames.

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