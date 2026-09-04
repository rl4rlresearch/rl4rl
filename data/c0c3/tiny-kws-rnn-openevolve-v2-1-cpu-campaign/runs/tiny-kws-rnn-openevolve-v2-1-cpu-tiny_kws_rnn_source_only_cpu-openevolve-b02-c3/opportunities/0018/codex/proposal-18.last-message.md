MECHANISM: Dual-timescale recurrent width refinement

HYPOTHESIS: A 92-unit dual-readout GRU processing the most recent 30 frames will achieve at least 85% validation accuracy while reducing total inference MACs below the qualified 94-unit model’s 787,244,360.

INTENDED_EDIT: Reduce the GRU hidden width from 98 to 92, resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly, and use the qualified 30-frame schedule.

EVIDENCE: The 94-unit, 30-frame dual-readout model achieved 86.99% accuracy, retaining 1.99 percentage points of margin after the preceding reduction from 96 units; another two-unit reduction directly tests the next structural cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 31, 0)
=======
        start = max(available_frames - 30, 0)
>>>>>>> REPLACE