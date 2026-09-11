MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 106-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 665,897,380 to approximately 654,490,640.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 109 to 106 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The adjacent 107-unit design achieved 86.38% validation accuracy at 665,897,380 MACs, and every tested 107–114-unit batch-64 GRU qualified, making 106 units the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(109, 8)
        self.endpoint_classifier = nn.Linear(109, 8)
=======
        self.gru = nn.GRU(20, 106, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(106, 8)
        self.endpoint_classifier = nn.Linear(106, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 106, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 106, device=device, dtype=dtype)
>>>>>>> REPLACE