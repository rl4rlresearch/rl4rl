MECHANISM: Single-unit recurrent width refinement

HYPOTHESIS: An 89-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 90-unit model’s 727,338,600.

INTENDED_EDIT: Reduce the GRU hidden width from 96 to 89 and resize its classifier, hidden state, and temporal summary while preserving the qualified 30-frame schedule and training procedure.

EVIDENCE: The 90-unit, 30-frame dual-readout GRU achieved 86.01% accuracy; its 1.01-point margin motivates a conservative one-unit reduction after the successful sequence of width refinements.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
        self.gru = nn.GRU(20, 89, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(178, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 89, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 89, device=device, dtype=dtype)
>>>>>>> REPLACE