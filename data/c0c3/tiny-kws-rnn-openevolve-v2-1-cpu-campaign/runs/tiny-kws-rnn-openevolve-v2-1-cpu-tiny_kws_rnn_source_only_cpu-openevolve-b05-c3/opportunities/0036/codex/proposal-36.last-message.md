MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 108-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 689,004,260 to approximately 677,401,920.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 113 to 108 units while preserving the qualified 20-frame schedule and batch-64 training procedure.

EVIDENCE: The adjacent 109-unit design achieved 86.99% validation accuracy at 689,004,260 MACs, leaving a 1.99-point margin; testing 108 units is the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 113, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(113, 8)
        self.endpoint_classifier = nn.Linear(113, 8)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
        self.endpoint_classifier = nn.Linear(108, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 113, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 113, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE