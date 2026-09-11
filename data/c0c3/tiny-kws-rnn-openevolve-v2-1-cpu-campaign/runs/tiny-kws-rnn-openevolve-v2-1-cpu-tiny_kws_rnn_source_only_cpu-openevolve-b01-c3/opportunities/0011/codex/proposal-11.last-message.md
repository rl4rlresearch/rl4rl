MECHANISM: Boundary-seeking recurrent width reduction

HYPOTHESIS: A 108-unit GRU with the proven batch-64 training regime and 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.3% versus the qualified 110-unit design.

INTENDED_EDIT: Reduce recurrent, summary-state, and classifier width from 112 to 108 units while preserving the validated schedule and training procedure.

EVIDENCE: The 110-unit design achieved 86.01% accuracy at 560,133,200 total MACs, improving accuracy over the wider 112-unit design; its 1.01-point margin supports testing another conservative two-unit reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE