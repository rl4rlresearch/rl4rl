MECHANISM: Dual-timescale recurrent width refinement

HYPOTHESIS: A 94-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 96-unit model’s 818,077,440.

INTENDED_EDIT: Reduce the GRU hidden width from 98 to 94 and resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly, while preserving the 30-frame schedule and training procedure.

EVIDENCE: The 96-unit dual-readout model reached 87.48% accuracy over 30 frames with 818,077,440 MACs, leaving 2.48 percentage points of margin; reducing width by two units previously remained qualified and substantially improved cost, motivating another two-unit reduction without removing temporal coverage.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
>>>>>>> REPLACE