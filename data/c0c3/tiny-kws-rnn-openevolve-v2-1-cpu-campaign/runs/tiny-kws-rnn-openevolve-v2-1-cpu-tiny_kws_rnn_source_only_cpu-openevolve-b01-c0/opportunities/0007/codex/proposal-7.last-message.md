MECHANISM: Width-for-temporal-coverage tradeoff

HYPOTHESIS: A 125-unit GRU processing 29 uniformly distributed frames will reach at least 85% validation accuracy while reducing exact dense inference MACs by approximately 0.9% versus the passing 128-unit, 28-step model.

INTENDED_EDIT: Reduce the recurrent and classifier width from 128 to 125, while increasing the causal frame schedule from 28 to 29 frames spanning the full recording.

EVIDENCE: The 125-unit, 28-step model narrowly missed the target at 84.908%, only 0.092 percentage points short; adding one temporal observation targets that deficit while the narrower recurrence still yields fewer MACs than the verified 128-unit, 28-step design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(29, available_frames)
>>>>>>> REPLACE