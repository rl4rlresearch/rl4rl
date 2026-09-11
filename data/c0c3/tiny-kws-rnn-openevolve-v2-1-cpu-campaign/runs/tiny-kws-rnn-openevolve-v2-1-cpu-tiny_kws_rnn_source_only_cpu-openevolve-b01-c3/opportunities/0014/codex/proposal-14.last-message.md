MECHANISM: One-unit boundary width trim

HYPOTHESIS: A 106-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 523,177,840.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and classifier width from 110 to 106 units while preserving the validated optimizer, augmentation, and even-index frame schedule.

EVIDENCE: The adjacent 107-unit design achieved 86.135% accuracy at 532,299,320 MACs; the prior 106-unit attempt timed out without producing contrary accuracy evidence, so retrying this smallest structural reduction directly tests the next unresolved cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
=======
        self.gru = nn.GRU(20, 106, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(106, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 106, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 106, device=device, dtype=dtype)
>>>>>>> REPLACE