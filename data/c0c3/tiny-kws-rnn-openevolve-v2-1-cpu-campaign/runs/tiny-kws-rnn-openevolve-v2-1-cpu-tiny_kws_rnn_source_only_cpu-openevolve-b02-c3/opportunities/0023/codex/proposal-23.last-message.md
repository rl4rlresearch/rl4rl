MECHANISM: Incremental temporal trimming of a qualified narrow GRU

HYPOTHESIS: An 88-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 30-frame model’s 698,265,920.

INTENDED_EDIT: Reduce hidden width from 90 to 88, resize the classifier and recurrent state, and omit the first three input frames.

EVIDENCE: The 88-unit, 30-frame model achieved 86.38% accuracy, while trimming the 98-unit model from 30 to 29 frames reduced accuracy by only 0.74 points and still qualified; applying the same single-frame trim to the narrower qualified design tests a larger structural cost reduction with 1.38 points of available margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 30, 0)
=======
        start = max(available_frames - 29, 0)
>>>>>>> REPLACE