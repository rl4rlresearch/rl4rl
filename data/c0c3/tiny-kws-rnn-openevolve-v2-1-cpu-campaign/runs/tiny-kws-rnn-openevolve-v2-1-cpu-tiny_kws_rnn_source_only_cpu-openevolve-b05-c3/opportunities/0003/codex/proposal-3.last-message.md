MECHANISM: Full-resolution narrow-state GRU

HYPOTHESIS: A 112-unit GRU retaining all 32 frames will preserve at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.

INTENDED_EDIT: Reduce the recurrent hidden width and classifier input width from 128 to 112 without temporal subsampling.

EVIDENCE: Both 16-step and 20-step schedules achieved only 84.66%, suggesting temporal evidence should be preserved; the full 32-step 128-unit model reached 87.24%, leaving a 2.24-point margin for a modest width reduction.

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