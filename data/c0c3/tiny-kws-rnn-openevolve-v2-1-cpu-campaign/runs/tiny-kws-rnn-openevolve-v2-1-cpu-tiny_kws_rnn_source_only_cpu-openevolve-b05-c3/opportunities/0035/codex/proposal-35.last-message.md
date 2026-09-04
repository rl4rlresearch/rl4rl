MECHANISM: Smaller-batch recurrent-width boundary probe

HYPOTHESIS: A 109-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 689,004,260.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 112 to 109 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The 110-unit batch-64 design qualified at 85.77% and 700,704,400 MACs; testing the adjacent 109-unit width is the most informative remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
        self.endpoint_classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(109, 8)
        self.endpoint_classifier = nn.Linear(109, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
>>>>>>> REPLACE