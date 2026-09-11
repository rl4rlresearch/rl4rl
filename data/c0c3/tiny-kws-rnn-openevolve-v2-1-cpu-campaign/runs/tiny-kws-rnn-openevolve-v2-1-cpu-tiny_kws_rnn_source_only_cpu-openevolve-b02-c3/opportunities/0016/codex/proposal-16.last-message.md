MECHANISM: Width–time tradeoff with dual-timescale readout

HYPOTHESIS: A 96-unit dual-readout GRU processing 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the current 98-unit, 29-frame model.

INTENDED_EDIT: Reduce recurrent width from 98 to 96, resize the mean-plus-final classifier, and restore one input frame for a predicted total of approximately 818.1M inference MACs.

EVIDENCE: The 98-unit dual-readout model achieved 85.77% at 30 frames, providing 0.77 percentage points of margin; exchanging two hidden units for that additional frame is the smallest width–time tradeoff that improves on the current 821.2M-MAC design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 29, 0)
=======
        start = max(available_frames - 30, 0)
>>>>>>> REPLACE