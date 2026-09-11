MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 105-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 654,490,640 to approximately 643,181,700.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 110 to 105 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The adjacent 106-unit design achieved 87.61% validation accuracy at 654,490,640 MACs, leaving a 2.61-point margin after every tested 106–114-unit batch-64 GRU qualified; 105 units is therefore the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
        self.endpoint_classifier = nn.Linear(110, 8)
=======
        self.gru = nn.GRU(20, 105, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(105, 8)
        self.endpoint_classifier = nn.Linear(105, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 105, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
>>>>>>> REPLACE