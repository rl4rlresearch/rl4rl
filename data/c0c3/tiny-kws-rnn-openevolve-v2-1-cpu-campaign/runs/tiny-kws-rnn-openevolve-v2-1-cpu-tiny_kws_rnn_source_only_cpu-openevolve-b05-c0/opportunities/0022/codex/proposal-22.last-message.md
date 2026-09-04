MECHANISM: Near-isocompute width–time reallocation

HYPOTHESIS: A 59-unit GRU processing the final 28 frames will recover validation accuracy to at least 85% while reducing total inference MACs by approximately 0.5% and recurrent steps from 29 to 28 versus the verified 58-unit, 29-frame model.

INTENDED_EDIT: Increase recurrent and summary width from 58 to 59 units, resize the classifier, and omit the first four frames.

EVIDENCE: The 58-unit, 28-frame model narrowly missed the target at 84.66%, while 29 frames achieved 85.28%; adding one hidden unit is the smallest capacity increase and still leaves the 28-step recurrent computation below the current model’s MAC count.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 59, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = 3 if available_frames > 4 else 0
=======
        start = 4 if available_frames > 5 else 0
>>>>>>> REPLACE